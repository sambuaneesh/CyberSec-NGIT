# remove the label column from the selection
if 'label' in df.columns:
    label_col = 'label'
elif 'Label' in df.columns:
    label_col = 'Label'
else:
    raise ValueError('Label column not found')
if label_col in df.columns:
    df = df.loc[:, df.columns != label_col]
