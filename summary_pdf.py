import psycopg2
import requests
import time
import pdfplumber
import textwrap
from io import BytesIO

# === SUMY ===
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer  

# === TRANSFORMERS (BART) ===
from transformers import pipeline, AutoTokenizer

# === CONFIG ===
DB_CONFIG = {
    "host":"1234.45.67.89",
}

CHUNK_SIZE = 2000
BATCH_LIMIT = 20       # smaller batch size per worker
BATCH_PAUSE_SECONDS = 2
MAX_WORDS = 1000

BART_MODEL = "sshleifer/distilbart-cnn-12-6"
BART_MAX_INPUT_TOKENS = 900

print("🚀 Loading DistilBART model...")
bart_tokenizer = AutoTokenizer.from_pretrained(BART_MODEL, model_max_length=1024)
bart_summarizer = pipeline(
    "summarization",
    model=BART_MODEL,
    tokenizer=bart_tokenizer,
    device=-1
)
print("✅ DistilBART loaded.")

# === Helpers ===
def download_pdf(url):
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    return BytesIO(resp.content)

def extract_text_from_pdf(pdf_file):
    text = ""
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text() or ""
            text += page_text + "\n"
    return text.strip()

def split_into_chunks(text, chunk_size=CHUNK_SIZE):
    return textwrap.wrap(text, chunk_size)

# --- SUMY ---
def sumy_summarize(text, sentence_count=5):
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    summarizer = LexRankSummarizer()
    summary = summarizer(parser.document, sentence_count)
    return " ".join(str(sentence) for sentence in summary)

# --- Token chunking ---
def split_into_token_chunks(text: str, tokenizer, max_tokens=BART_MAX_INPUT_TOKENS):
    enc = tokenizer(text, add_special_tokens=False, return_attention_mask=False, return_token_type_ids=False)
    ids = enc["input_ids"]
    chunks = []
    for i in range(0, len(ids), max_tokens):
        piece = ids[i:i+max_tokens]
        chunk_text = tokenizer.decode(piece, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        if chunk_text.strip():
            chunks.append(chunk_text)
    return chunks

# --- BART ---
def bart_summarize(text, max_len=200, min_len=60):
    if not text.strip():
        return ""

    chunks = split_into_token_chunks(text, bart_tokenizer, max_tokens=BART_MAX_INPUT_TOKENS)
    print(f"🧩 BART: split into {len(chunks)} token-chunks.")

    partial_summaries = []
    for idx, c in enumerate(chunks, 1):
        try:
            input_words = len(c.split())
            out_max = min(max_len, max(32, int(min(input_words * 0.9, 200))))
            out_min = min(min_len, max(10, int(out_max * 0.5)))
            if out_min >= out_max:
                out_min = max(10, out_max - 10)

            print(f"🤖 BART summarizing chunk {idx}/{len(chunks)} (~{input_words} words)...")
            summary = bart_summarizer(
                c,
                max_length=out_max,
                min_length=out_min,
                do_sample=False,
                truncation=True
            )[0]["summary_text"]

            partial_summaries.append(summary)
        except Exception as e:
            print(f"⚠️ BART failed on chunk {idx}: {e}")
            partial_summaries.append(" ".join(c.split()[:100]))

    return " ".join(partial_summaries)

# --- Hybrid Summarization ---
def summarize_text(text):
    chunks = split_into_chunks(text)
    print(f"📑 Split into {len(chunks)} chunks for Sumy...")

    intermediate_summaries = []
    for idx, chunk in enumerate(chunks, start=1):
        try:
            per_chunk_sentences = max(3, (MAX_WORDS // 20) // len(chunks))
            summary = sumy_summarize(chunk, sentence_count=per_chunk_sentences)
            if summary:
                intermediate_summaries.append(summary)
        except Exception as e:
            print(f"⚠️ Sumy error in chunk {idx}: {e}")
            intermediate_summaries.append(chunk[:500])

    combined_sumy = " ".join(intermediate_summaries)
    print("✂️ Sumy intermediate summary ready.")

    return bart_summarize(combined_sumy, max_len=200, min_len=60)

# === DB ===
def get_db_connection():
    return  psycopg2.connect()
# psycopg2.connect(**DB_CONFIG)

def fetch_batch(cursor, batch_limit):
    cursor.execute(f"""
        UPDATE extracted_judgments
        SET is_synced = TRUE
        WHERE id IN (
            SELECT id
            FROM extracted_judgments
            WHERE (summary IS NULL OR summary = '')
              AND (is_synced IS NULL OR is_synced = FALSE)
            ORDER BY id ASC
            LIMIT %s
            FOR UPDATE SKIP LOCKED
        )
        RETURNING id, judgment_url, "extractedText";
    """, (batch_limit,))
    return cursor.fetchall()


def process_judgments():
    conn = get_db_connection()
    cursor = conn.cursor()

    while True:
        # 🚀 Atomically claim a batch of rows
        records = fetch_batch(cursor, BATCH_LIMIT)
        conn.commit()  # make sure locks are released for other workers

        if not records:
            print("✅ No more records to process.")
            break

        print(f"🚀 Picked {len(records)} records.")
        for judgment_id, url, extracted_text in records:
            try:
                print(f"\n📥 Processing ID {judgment_id}")
                if extracted_text and extracted_text.strip():
                    text = extracted_text.strip()
                    print(f"✅ Using extractedText from DB ({len(text)} chars)")
                else:
                    if not url:
                        print(f"⚠️ No URL for ID {judgment_id}, skipping.")
                        continue

                    print(f"⬇️ Downloading PDF from {url}")
                    try:
                        pdf_file = download_pdf(url)
                        text = extract_text_from_pdf(pdf_file)
                        print(f"✅ Extracted {len(text)} chars from PDF")
                    except Exception as e:
                        print(f"❌ Could not download PDF for ID {judgment_id}: {e}")
                        # Remove broken URL so it won't be retried
                        cursor.execute("""
                            DELETE FROM extracted_judgments
                            WHERE judgment_url = %s
                        """, (url,))
                        conn.commit()
                        continue  # skip to next record

                if not text:
                    print(f"⚠️ No text for ID {judgment_id}, skipping.")
                    continue

                summary = summarize_text(text)

                if summary:
                    cursor.execute("""
                        UPDATE extracted_judgments
                        SET summary = %s
                        WHERE id = %s
                    """, (summary, judgment_id))
                    conn.commit()
                    print(f"✅ Saved summary for ID {judgment_id}")

            except Exception as e:
                print(f"❌ Error processing ID {judgment_id}: {e}")
                conn.rollback()
                # reset is_synced so another worker can retry
                cursor.execute("""
                    UPDATE extracted_judgments
                    SET is_synced = FALSE
                    WHERE id = %s
                """, (judgment_id,))
                conn.commit()

        print(f"⏸ Pausing {BATCH_PAUSE_SECONDS}s before next fetch...")
        time.sleep(BATCH_PAUSE_SECONDS)

    cursor.close()
    conn.close()
    print("🎉 All records processed.")

if __name__ == "__main__":
    process_judgments()