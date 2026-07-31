from scraper import main as scraper_main
from cleaner import main as cleaner_main
from database import main as database_main
from queries import main as queries_main


def main() -> None:
    """
    Execute the complete ETL pipeline.
    """

    print("=" * 60)
    print("BOOKS ETL PIPELINE")
    print("=" * 60)

    print("\nStep 1: Extract")
    scraper_main()

    print("\nStep 2: Transform")
    cleaner_main()

    print("\nStep 3: Load")
    database_main()

    print("\nStep 4: Analyze")
    queries_main()

    print("\n" + "=" * 60)
    print("ETL PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 60)


if __name__ == "__main__":
    main()