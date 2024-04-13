import numpy as np
import pandas as pd
from docx import Document
from utilities.utility_ComputationalModeling import bayes_factor


def mean_AIC_BIC(df):
    print(f"AIC: {df['AIC'].mean()}")
    print(f"BIC: {df['BIC'].mean()}")


# Function to create Bayes factor matrix
def create_bayes_matrix(simulations, file_name):

    def format_large_numbers(num):
        if num > 1000:
            exponent = np.floor(np.log10(num))
            mantissa = num / 10 ** exponent
            return f"{mantissa:.3f} * 10^{exponent:.0f}"
        if num < 0.001:
            return f"<0.001"
        else:
            return f"{num:.3f}"

    # Function to add a dataframe to the document
    def add_df_to_doc(df, title):
        doc = Document()
        doc.add_heading(title, level=1)
        table = doc.add_table(df.shape[0] + 1, df.shape[1] + 1)  # Add an extra column for the row names

        # Add the column names
        for j in range(df.shape[-1]):
            table.cell(0, j + 1).text = df.columns[j]  # Shift the column names to the right

        # Add the row names and data
        for i in range(df.shape[0]):
            table.cell(i + 1, 0).text = df.index[i]  # Add the row name
            for j in range(df.shape[-1]):
                table.cell(i + 1, j + 1).text = str(df.values[i, j])  # Shift the data to the right

        doc.add_paragraph("\n")

        # Save the document
        doc.save(f"./data/DataFitting/BayesFactor/{title}.docx")

    model_names = list(simulations.keys())
    bayes_matrix = pd.DataFrame(index=model_names, columns=model_names)
    for null_model in model_names:
        for alternative_model in model_names:
            if null_model != alternative_model:
                bayes_matrix.loc[null_model, alternative_model] = bayes_factor(simulations[null_model], simulations[alternative_model])

    bayes_matrix = bayes_matrix.T

    add_df_to_doc(bayes_matrix.applymap(format_large_numbers), file_name)
    return bayes_matrix.applymap(format_large_numbers)


def process_chosen_prop(results, data, sub=False):
    results['best_process_chosen'] = results['best_process_chosen'].apply(lambda x: x if isinstance(x, list) else eval(x))
    process_chosen = results['best_process_chosen'].explode()
    pd.options.mode.chained_assignment = None
    data['best_process_chosen'] = process_chosen.values
    # check if there is a TrialType column
    if 'TrialType' not in data.columns:
        mappping = {0: 'AB', 1: 'CD', 2: 'CA', 3: 'CB', 4: 'AD', 5: 'BD'}
        data['TrialType'] = data['SetSeen.'].map(mappping)

    if sub:
        process_chosen_df = data.groupby(['Subnum', 'TrialType'])['best_process_chosen'].value_counts(normalize=True).reset_index()
    else:
        process_chosen_df = data.groupby('TrialType')['best_process_chosen'].value_counts(normalize=True).unstack().fillna(0)

    return data, process_chosen_df






