from openai import OpenAI
from ..flare import *
import pytesseract
from pdf2image import convert_from_path
import ast
import pandas as pd
import numpy as np

class Extractor:
    def __init__(self, api_key):
        self.client = OpenAI(api_key=api_key)
        self.api_key = api_key
        self.raw_extract = """In the given fund facts statement text, please extract the following information:
        - Name of the fund manager
        - Name of the portfolio manager
        - Date of the document
        - Date ETF Started
        - As at Date of Investments or assets information
        - Name of the fund in subject
        - Name of the fund series or class
        - ETF ticker symbol
        - Management Expense Ratio (MER)
        - Distributions, ex: Monthly, Quarterly etc
        - Minimum or initial Investment required.
        - Total fund value
        - listed stock exchange code
        - Currency
        - Average daily volume
        - Number of days traded
        - Market price range
        - Net Asset Value(NAV) price range, return decimals without the dollar sign.
        - Average bid-ask spread in percentage
        - Risk rating of the fund (risk level or volatility rating of the ETF)
        - Investment mix with their respective sizes
        - Top N investments with their respective sizes.
        - Year-by-year returns
        - Best return
        - worst return
        - Average annual compounded return
        - Trading Expense Ratio (TER)
        - Annual Management Fee

        It is imperative that you follow the following format for output:
        Output Format:
        {'fund_manager': string,
        'portfolio_manager': string,
        'document_date': list[yyyy-mm-dd],
        'etf_start_date': list[yyyy-mm-dd],
        'assets_date': list[yyyy-mm-dd],
        'fundname': list[string],
        'fundseries': list[string],
        'etf_ticker': list[string],
        'management_expense_ratio': list[decimal],
        'distributions': list[string],
        'minimum_investment': decimal,
        'totalfundvalue' : decimal,
        'stock_exchange': string,
        'currency': string,
        'avg_daily_volume': list[integer],
        'number_of_days_traded': list[integer],
        'market_price': list[decimal1, decimalN],
        'nav':[amount1, amountN]:list[decimal1, decimalN],
        'avg_bid-ask_spread': decimal
        'riskrating': low to medium, medium, high etc: string,
        'assets_mix': [(asset1, size)....(assetN, size)] : list[(string, string or decimal)],
        'top_investments': [(investment1, size)....(investmentsN, size)] : list[(string, string or decimal)],
        'year_by_year_returns':[(year1,return%),(yearN,return%)]: list[(int, string or decimal)]
        'best_return': [date,return] : list[date, decimal],
        'worst_return': [date,return] : list[date, decimal],
        'average_return': list[decimal],
        'trading_expense_ratio': list[decimal],
        'annual_management_fee': list[decimal]}

        Additional rules for extraction:
        - Make sure to return only if you find the information in the text
        - Return all dates in 'yyyy-mm-dd' string format within the list
        - For totalfundvalue and minimum_investment, always make sure to return the amount in decimals. Ex: $600 MIllion should be returned as 600000000.00
        - For stock_exchange and currency, only return the exchange code (TSX, NYSE etc) and ISO currency code (CAD, USD etc)
        - For all percentages found in the pdf, it's critical to not change it to decimals. For example, 1.06% should be kept as 1.06
        - For Distributions, make sure to only return the period in the first index, put any additional notes in the following indexes. Ex: ['Monthly', 'additional notes here']
        - I have included the types just for your reference, do not include them in the output
        - If the information is not found, set all not-found values to None and still return the dictionary in the given format.
        - It is critical to NOT include any other text.
        - DO NOT include any footnotes.
        - It is critical to always start the output with an open curly bracket '{'.


        Fund Facts Statement start:
        """

        self.dict_reformat = """ You are given an invalid dictionary string that I was unable to convert to dictionary object using ast.literal_eval. I need your help to update the
        string so that I can convert this into a dictionary object.
        - Fix any format issues that might be causing an error.
        - Make sure to change any NULL or None to null.
        - Do not include any other text, you do NOT need to say "Sure! Here is the cleaned dictionary in the requested format:"
        - Your output will be passed as is to ast.literal_eval
        - You keep returning additional text outside the dictionary. Your output should be limited to the dictionary only.
        - Your output should start with "{" and end with "}". Please do not even say "Sure! Here is your output"
        Text start:
        """


    def extract_text_from_pdf_pytesseract(pdfs_to_extract, page_limit):
        # Initialize a dictionary to store the extracted text
        pdf_data_dict = {}

        # Iterate over the list of PDFs
        for pdf in pdfs_to_extract:
            pdf_data_dict[pdf] = {}
            page_counter = 0

            images = convert_from_path(pdf)
            for page in images[:page_limit]:
                text = pytesseract.image_to_string(page)
                pdf_data_dict[pdf][page_counter] = text
                page_counter += 1

        # Get the number of pages in each PDF
        page_counts = {key: len(inner_dict) for key, inner_dict in pdf_data_dict.items()}

        return pdf_data_dict, page_counts

    def create_model_response_flare(self, prompt, text, page):
        while True:
            try:
                original_answer, annotated_answer, final_answer = flare(
                    prompt, Fake_Retriever(text), self.api_key, verbose=False)
                break
            except:
                print(f"Repeating FLARE for page {page}")
                continue
        return final_answer.choices[0].message.content

    def create_model_response(self, prompt, text):
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are an AI assistant that helps extract structured information from a given unstructured text."},
                {"role": "user", "content": f'{prompt}\n{text}'},
            ]
        )
        return response.choices[0].message.content

    def create_extract(self, pdf_data_dict, method='regular'):
        # dictionary to save the final outputs
        output_dict = {}

        # iterate through each pdf
        for pdfname in pdf_data_dict.keys():
            output_dict[pdfname] = {}
            # for each page in the pdf
            for page in pdf_data_dict[pdfname].keys():
                if method == 'regular':
                    raw_model_output = self.create_model_response_flare(self.raw_extract, pdf_data_dict[pdfname][page], page)
                elif method == 'flare':
                    raw_model_output = self.create_model_response(self.raw_extract, pdf_data_dict[pdfname][page], page)
                else:
                    print("Please specify method as 'regular' or 'flare'")
                    return

                try:
                    output_dict[pdfname][page] = ast.literal_eval(raw_model_output)
                except Exception as e:
                    try:
                        # Create a new model response and attempt conversion again
                        cleaned_model_output = self.create_model_response(self.dict_reformat, raw_model_output)
                        output_dict[pdfname][page] = ast.literal_eval(cleaned_model_output)
                    except Exception as e:

                        # Step 1: Extract the dictionary part
                        start_index = cleaned_model_output.find('{')
                        end_index = cleaned_model_output.rfind('}') + 1
                        dict_string = cleaned_model_output[start_index:end_index]

                        # Step 2: Convert the string to a dictionary
                        # Replace % with a suitable numeric representation (e.g., 2.40% -> 2.40)
                        dict_string = re.sub(r'(\d+(\.\d+)?)%', r'\1', dict_string)
                        # Replace null with None
                        dict_string = dict_string.replace('null', 'None')
                        # Remove dollar signs and commas from monetary values
                        dict_string = re.sub(r'\$\s*([\d,]+\.\d+|\d+)', lambda m: m.group(1).replace(',', ''), dict_string)
                        # Convert lists with dollar values and commas
                        dict_string = re.sub(r'\[\s*([$,\d.]+)\s*\]', lambda m: f"[{m.group(1).replace(',', '')}]", dict_string)
                        try:
                            output_dict[pdfname][page] = ast.literal_eval(dict_string)
                        except Exception as e:
                            print(f"Failed to convert string to dictionary for: {pdfname} page: {page}")
                            return
        return output_dict


    def dict_to_df(self, data_dict):
        """
        Converts a dictionary to a DataFrame.

        Args:
            data_dict (Dict[str, Dict[int, Dict[str, Any]]]): The dictionary to be converted.

        Returns:
            The converted Pandas DataFrame.
        """
        # Create a list to store DataFrame rows
        rows = []

        # Iterate over the dictionary and append rows to the list
        for filename, pages in data_dict.items():
            for page, details in pages.items():
                row = {"filename": filename, "page": page, **details}
                rows.append(row)

        # Convert the list of rows to a DataFrame
        df = pd.concat([pd.DataFrame([i]) for i in rows], ignore_index=True)


        # clean df
        df = df.replace(
            ["Unknown", "None", "['Unknown']", "[]",
                "[]", "[None]", "['N/A*']", "['N/A']", 'N/A'],
            None,
        )
        # remove empty lists
        df = df.applymap(lambda x: None if isinstance(
            x, list) and not x else x)

        # remove lists with 'None' string values
        df = df.applymap(
            lambda x: (
                None
                if isinstance(x, list) and all(str(item).lower() == "none" for item in x)
                else x
            )
        )
        return df

    def extract_sequences(self, df):
        """
        Extracts sequences from a DataFrame and stores them in another DataFrame and a dictionary.

        Parameters:
            df: The input Pandas DataFrame.

        Returns:
        Tuple: A tuple [pd.DataFrame, Dict[str, pd.DataFrame]] containing a DataFrame with the 
        extracted sequences and a dictionary mapping each filename to a DataFrame with its extracted sequences.
        """

        # Get the distinct filenames
        distinct_filenames = df["filename"].drop_duplicates().tolist()

        # Initialize an empty DataFrame for df_sequences
        df_sequences = pd.DataFrame(
            columns=["row_numbers", "value_counts", "in_between_pages"]
        )

        dict_sequences = {}
        for filename in distinct_filenames:
            # Filter the DataFrame by the current filename
            filtered_df = df[df["filename"] == filename]

            # Count the number of populated columns for each row
            populated_columns_count = filtered_df.notnull().sum(axis=1)

            # Create a temporary DataFrame with the row numbers and their corresponding counts
            df_counts = pd.DataFrame(
                {
                    "row_numbers": populated_columns_count.index,
                    "value_counts": populated_columns_count.values,
                }
            )

            # Split the DataFrame into multiple smaller DataFrames to reduce sequence search times
            split_dfs = self.split_dataframe(df_counts)

            # Run the sequence search on all the smaller DataFrames
            sequenced_df_list = [self.sequence_search(
                df) for df in split_dfs if not df.empty]

            # Concatenate the smaller DataFrames into a single DataFrame and reset its index
            df_temp = pd.concat(sequenced_df_list).reset_index(drop=True)

            # Drop duplicate rows, keeping only the last occurrence
            df_temp.drop_duplicates(subset="row_numbers",
                                    keep="last", inplace=True)

            # Add the sequences to the dictionary
            dict_sequences[filename] = df_temp

            # Append df_temp to df_sequences
            df_sequences = pd.concat([df_sequences, df_temp])

        return df_sequences, dict_sequences


    def split_dataframe(self, df):
        """
        Splits a DataFrame into multiple smaller DataFrames based on the maximum value count.

        Args:
            df: The input Pandas DataFrame.

        Returns:
            A list of the split Pandas DataFrames.
        """

        # Find the maximum value count
        max_value_count = df["value_counts"].max()

        # Find the indices where value_count equals max_value_count
        split_indices = df.index[df["value_counts"]
                                 == max_value_count].tolist()

        # Add the last index of the dataframe to the list
        split_indices.append(len(df))

        # Initialize a list to hold all the split dataframes
        dfs = []

        # The start index for the first split dataframe is 0
        start_index = 0

        # Iterate over the split indices
        for end_index in split_indices:
            # Split the dataframe and append it to the list
            dfs.append(df.iloc[start_index:end_index])

            # The start index for the next split dataframe is the end index of the current split dataframe
            start_index = end_index

        return dfs


    def sequence_search(self, df):
        """
        Searches for sequences in a DataFrame.

        Args:
            df: The input Pandas DataFrame.

        Returns:
            The DataFrame with the found sequences.
        """

        # Get the maximum value count
        max_sequence_count = df["value_counts"].max()

        # Get all rows where sequence count is equal to max_sequence_count
        df_max = df[df["value_counts"] == max_sequence_count]

        # Count all rows between each "max" row
        df_max["in_between_pages"] = df_max["row_numbers"].diff() - 1
        # print(df_max)

        # Check if there are more than 4 pages (3+itself) in between or after the last "highest" number page
        large_gaps = df_max[df_max["in_between_pages"] > 3]

        for index, row in large_gaps.iterrows():
            start = (
                df_max.loc[index - 1, "row_numbers"]
                if index - 1 in df_max.index
                else df["row_numbers"].min()
            )
            end = row["row_numbers"]
            gap_df = df[(df["row_numbers"] > start) & (df["row_numbers"] < end)]
            if not gap_df.empty:
                df_max = pd.concat([df_max, self.sequence_search(gap_df)])

        # Check if there are more than four pages left after the last "highest" number page
        last_row_number = df_max.iloc[-1]["row_numbers"]
        remaining_df = df[df["row_numbers"] > last_row_number]

        # again using 4 pages as cutoff - after the last recorded page (this will catch 5 page statements)
        if not remaining_df.empty and len(remaining_df) > 4:
            df_max = pd.concat([df_max, self.sequence_search(remaining_df)])

        return df_max

    def combine_sequences_to_create_extract(self, df, df_sequences, cols_to_fill=None):
        """
        Combines sequences to create an extract.

        Args:
            df: The input Pandas DataFrame.
            df_sequences: The Pandas DataFrame with the sequences.

        Returns:
           The DataFrame with the combined sequences.
        """

        # Merge sequences to main DataFrame
        df_with_sequences = df.reset_index().merge(
            df_sequences, left_on="index", right_on="row_numbers", how="left"
        )

        # Add title_page column
        df_with_sequences["title_page"] = df_with_sequences["row_numbers"].apply(
            lambda x: 1 if pd.notnull(x) else 0
        )

        # Cumulative sum on the title page
        df_with_sequences["document_id"] = (df_with_sequences["title_page"] == 1).cumsum()

        for col in cols_to_fill:
            df_with_sequences[col] = df_with_sequences.groupby("document_id")[col].bfill()

        # Drop rows where title_page = 0
        df_with_sequences = df_with_sequences[df_with_sequences["title_page"] == 1]

        # Drop the document_id column
        df_with_sequences.drop("document_id", axis=1, inplace=True)


        return df_with_sequences