from pypdf import PdfReader
import glob
import os

for pdf_file in glob.glob('./downloads/*.pdf'):
    try:
        #print(f'Reading file {pdf_file}')
        reader = PdfReader(pdf_file)
        number_of_pages = len(reader.pages)
        # page = reader.pages[0]
        # text = page.extract_text()
        #print(f'PDF {pdf_file} has {number_of_pages} pages')
    except:
        print(f'Unable to read file {pdf_file}')
        os.remove(pdf_file)