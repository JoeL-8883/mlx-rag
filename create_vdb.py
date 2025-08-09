import argparse
from vdb import vdb_from_pdf, vdb_from_pdf_dir
import os
import glob

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a Vector DB from a PDF file")
    # Input
    parser.add_argument(
        "--pdf",
        help="The path to the input PDF file",
        default="flash_attention.pdf",
    )

    # Input - pdf directory
    parser.add_argument(
        "--pdf_dir",
        type=str,
        help="The directory containing the PDF files",
    )

    # Output
    parser.add_argument(
        "--vdb",
        type=str,
        default="vdb.npz",
        help="The path to store the vector DB",
    )
    
    args = parser.parse_args()
    
    # Get a list of PDF files within the specified directory
    # glob matches all files that end in .pdf
    if args.pdf_dir:
        pdf_files = glob.glob(os.path.join(args.pdf_dir, "*.pdf"))
        m = vdb_from_pdf_dir(pdf_files)
        m.savez(args.vdb)
        if not pdf_files:
            raise ValueError(f"No PDF files found in the directory: {args.pdf_dir}")
        print(f"Found {len(pdf_files)} PDF files in the directory: {args.pdf_dir}")
    elif args.pdf:
        m = vdb_from_pdf(args.pdf)
        m.savez(args.vdb)
    else:
        raise ValueError("Please provide either a PDF file or a directory containing PDF files.")
