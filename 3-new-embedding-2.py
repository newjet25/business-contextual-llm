from typing import List
import os, json, argparse
import lancedb
from docling.chunking import HybridChunker
from docling.document_converter import DocumentConverter
from dotenv import load_dotenv
from lancedb.embeddings import get_registry
from lancedb.pydantic import LanceModel, Vector
from utils.tokenizer import OpenAITokenizerWrapper

load_dotenv()

# --------------------------------------------------------------
# Parse CLI args
# --------------------------------------------------------------
parser = argparse.ArgumentParser(description="cobol-token -> LanceDB pipeline")
parser.add_argument("--force-rebuild", action="store_true", help="Rebuild table & re-embed documents")
args = parser.parse_args()

# --------------------------------------------------------------
# Setup
# --------------------------------------------------------------
db = lancedb.connect("data/lancedb")
tokenizer = OpenAITokenizerWrapper()
MAX_TOKENS = 8191
func = get_registry().get("openai").create(name="text-embedding-3-large")

TABLE_NAME = "cobol-token"

# --------------------------------------------------------------
# Schema
# --------------------------------------------------------------
class ChunkMetadata(LanceModel):
    filename: str | None
    page_numbers: List[int] | None
    title: str | None


class Chunks(LanceModel):
    text: str = func.SourceField()
    vector: Vector(func.ndims()) = func.VectorField()  # auto-embedding
    metadata: str


# --------------------------------------------------------------
# Create or load table
# --------------------------------------------------------------
if TABLE_NAME in db.table_names() and not args.force_rebuild:
    print(f"🔹 Using existing table: {TABLE_NAME}")
    table = db.open_table(TABLE_NAME)
else:
    mode = "overwrite" if args.force_rebuild else "create"
    print(f"🆕 Building table ({mode})")
    table = db.create_table(TABLE_NAME, schema=Chunks, mode=mode)

    # Process + embed only if fresh/forced
    converter = DocumentConverter()
    result = converter.convert("cobol_tutorial.pdf")

    chunker = HybridChunker(
        tokenizer=tokenizer,
        max_tokens=MAX_TOKENS,
        merge_peers=True,
    )
    chunks = list(chunker.chunk(dl_doc=result.document))

    processed_chunks = [
        {
            "text": chunk.text,
            "metadata": json.dumps({
                "filename": chunk.meta.origin.filename,
                "page_numbers": sorted({
                    prov.page_no
                    for item in chunk.meta.doc_items
                    for prov in item.prov
                }) or None,
                "title": chunk.meta.headings[0] if chunk.meta.headings else None,
            }),
        }
        for chunk in chunks
    ]

    print(f"👉 Adding {len(processed_chunks)} chunks (embedding with OpenAI)")
    table.add(processed_chunks)


# --------------------------------------------------------------
# Query
# --------------------------------------------------------------
print(f"📊 Rows in table: {table.count_rows()}")
result = table.search("what's docling?").limit(3)
print(result.to_pandas())
