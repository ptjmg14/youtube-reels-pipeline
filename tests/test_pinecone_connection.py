
from youtube_reels.config import Settings
from youtube_reels.pinecone_store import PineconeStore


def test_pinecone_connection():
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass
    settings = Settings.from_env()
    
    if not settings.pinecone_api_key:
        print("❌ PINECONE_API_KEY not found in .env")
        return

    print(f"Testing Pinecone connection to index: {settings.pinecone_index_name}...")
    try:
        store = PineconeStore(settings)
        print("✅ Successfully initialized PineconeStore.")
        
        # Test a simple search (might be empty but shouldn't crash)
        results = store.search("test query", limit=1)
        print(f"✅ Search executed successfully. Found {len(results)} results.")
        
    except Exception as e:  # noqa: BLE001 - test probe prints the failure
        print(f"❌ Pinecone test failed: {e}")

if __name__ == "__main__":
    test_pinecone_connection()
