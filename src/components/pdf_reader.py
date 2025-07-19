# src/components/pdf_reader.py

import fitz  # PyMuPDF
import os

class PDFReader:
   
    def __init__(self, pdf_path: str):
        
        self.pdf_path = pdf_path
        self._doc = None # To hold the fitz.Document object

    def _open_pdf(self):
        """
        Opens the PDF document. Internal helper method.
        """
        if not os.path.exists(self.pdf_path):
            raise FileNotFoundError(f"The PDF file was not found: {self.pdf_path}")
        try:
            self._doc = fitz.open(self.pdf_path)
        except fitz.EmptyFileError:
            raise ValueError(f"The PDF file is empty or corrupted: {self.pdf_path}")
        except Exception as e:
            raise IOError(f"Failed to open PDF file '{self.pdf_path}': {e}")

    def _close_pdf(self):
        """
        Closes the PDF document if it's open. Internal helper method.
        """
        if self._doc:
            self._doc.close()
            self._doc = None # Clear the reference

    def get_text(self) -> str:
        """
        Extracts all text content from the PDF file.

        Returns:
            str: A single string containing all the extracted text from the PDF.
                 Returns an empty string if no text can be extracted or
                 if there's an issue with the file after opening.
        Raises:
            FileNotFoundError: If the PDF file does not exist.
            ValueError: If the PDF file is empty or corrupted.
            IOError: For other errors related to opening or reading the PDF.
        """
        extracted_text = ""
        try:
            self._open_pdf() # Ensure the document is open
            for page_num in range(self._doc.page_count):
                page = self._doc.load_page(page_num)
                extracted_text += page.get_text()
        finally:
            self._close_pdf() # Ensure the document is always closed

        return extracted_text.strip() # .strip() to remove leading/trailing whitespace

    def get_page_count(self) -> int:
        """
        Returns the total number of pages in the PDF document.

        Returns:
            int: The number of pages.
        Raises:
            FileNotFoundError: If the PDF file does not exist.
            ValueError: If the PDF file is empty or corrupted.
            IOError: For other errors related to opening the PDF.
        """
        try:
            self._open_pdf()
            return self._doc.page_count
        finally:
            self._close_pdf()

    def get_page_text(self, page_number: int) -> str:
        """
        Extracts text from a specific page of the PDF.

        Args:
            page_number (int): The 0-based index of the page to extract text from.

        Returns:
            str: The text content of the specified page.
        Raises:
            FileNotFoundError: If the PDF file does not exist.
            ValueError: If the PDF file is empty, corrupted, or page number is out of range.
            IOError: For other errors related to opening or reading the PDF.
            IndexError: If the page_number is out of the valid range.
        """
        try:
            self._open_pdf()
            if not (0 <= page_number < self._doc.page_count):
                raise IndexError(f"Page number {page_number} is out of range. PDF has {self._doc.page_count} pages.")
            page = self._doc.load_page(page_number)
            return page.get_text().strip()
        finally:
            self._close_pdf()

if __name__ == "__main__":
    # This block is for testing the PDFReader class independently.
    # You'll need a 'sample.pdf' or you can uncomment the reportlab part
    # to generate one (requires 'pip install reportlab').
    try:
        from reportlab.pdfgen import canvas
        if not os.path.exists("sample.pdf"):
            c = canvas.Canvas("sample.pdf")
            c.drawString(100, 750, "Hello, this is a sample PDF.")
            c.drawString(100, 730, "It contains some example text.")
            c.drawString(100, 710, "Page 1.")
            c.showPage()
            c.drawString(100, 750, "This is page 2.")
            c.drawString(100, 730, "More text here.")
            c.save()
            print("Created 'sample.pdf' for testing.")
    except ImportError:
        print("Install 'reportlab' (pip install reportlab) to create a sample PDF programmatically.")
        print("Please ensure 'sample.pdf' exists in the same directory for the example to work.")

    test_pdf_path = "sample.pdf"
    if os.path.exists(test_pdf_path):
        try:
            reader = PDFReader(test_pdf_path)
            print(f"\n--- Testing PDFReader with '{test_pdf_path}' ---")
            print(f"Total Pages: {reader.get_page_count()}")
            print("\nFull Text:")
            print(reader.get_text())
        except Exception as e:
            print(f"Error during PDFReader test: {e}")
    else:
        print(f"'{test_pdf_path}' not found for testing PDFReader.")