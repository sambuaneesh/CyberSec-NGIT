list1 = ['Dst Port', 'Protocol', 'Fwd Pkt Len Max', 'Fwd Pkt Len Min',
         'Fwd Pkt Len Mean', 'Fwd Pkt Len Std', 'Bwd Pkt Len Max',
         'Bwd Pkt Len Min', 'Bwd Pkt Len Mean', 'Bwd Pkt Len Std', 'Bwd IAT Tot',
         'Bwd IAT Std', 'Bwd IAT Max', 'Fwd Pkts/s', 'Bwd Pkts/s', 'Pkt Len Min',
         'Pkt Len Max', 'Pkt Len Mean', 'Pkt Len Std', 'PSH Flag Cnt',
         'ACK Flag Cnt', 'URG Flag Cnt', 'Down/Up Ratio', 'Pkt Size Avg',
         'Fwd Seg Size Avg', 'Bwd Seg Size Avg', 'Init Fwd Win Byts',
         'Init Bwd Win Byts', 'Fwd Act Data Pkts', 'Fwd Seg Size Min']

list2 = ['Protocol', 'Flow Duration', 'Fwd Pkt Len Std', 'Bwd Pkt Len Min',
         'Bwd Pkt Len Std', 'Flow IAT Std', 'Flow IAT Max', 'Fwd IAT Tot',
         'Fwd IAT Mean', 'Fwd IAT Max', 'Fwd IAT Min', 'Bwd IAT Tot',
         'Bwd IAT Mean', 'Bwd IAT Std', 'Bwd IAT Max', 'Bwd IAT Min',
         'Fwd PSH Flags', 'Pkt Len Min', 'SYN Flag Cnt', 'PSH Flag Cnt',
         'Init Fwd Win Byts', 'Fwd Seg Size Min', 'Active Mean', 'Active Std',
         'Active Max', 'Active Min', 'Idle Mean', 'Idle Std', 'Idle Max',
         'Idle Min']
list3 = ['Dst Port', 'Tot Bwd Pkts', 'TotLen Fwd Pkts', 'Fwd Pkt Len Max',
         'Fwd Pkt Len Mean', 'Fwd Pkt Len Std', 'Bwd Pkt Len Max',
         'Bwd Pkt Len Mean', 'Bwd Pkt Len Std', 'Flow IAT Std', 'Fwd IAT Tot',
         'Fwd Header Len', 'Bwd Header Len', 'Bwd Pkts/s', 'Pkt Len Max',
         'Pkt Len Mean', 'Pkt Len Std', 'Pkt Len Var', 'PSH Flag Cnt',
         'ACK Flag Cnt', 'Down/Up Ratio', 'Pkt Size Avg', 'Fwd Seg Size Avg',
         'Bwd Seg Size Avg', 'Subflow Fwd Byts', 'Subflow Bwd Pkts',
         'Init Fwd Win Byts', 'Init Bwd Win Byts', 'Fwd Act Data Pkts',
         'Fwd Seg Size Min']
list4 = ['Dst Port', 'Protocol', 'Tot Bwd Pkts', 'TotLen Fwd Pkts',
         'Fwd Pkt Len Max', 'Fwd Pkt Len Mean', 'Fwd Pkt Len Std',
         'Bwd Pkt Len Mean', 'Flow Byts/s', 'Fwd IAT Tot', 'Fwd Header Len',
         'Bwd Pkts/s', 'Pkt Len Max', 'Pkt Len Mean', 'Pkt Len Std',
         'Pkt Len Var', 'RST Flag Cnt', 'PSH Flag Cnt', 'ACK Flag Cnt',
         'ECE Flag Cnt', 'Down/Up Ratio', 'Pkt Size Avg', 'Fwd Seg Size Avg',
         'Bwd Seg Size Avg', 'Subflow Fwd Byts', 'Subflow Bwd Pkts',
         'Init Fwd Win Byts', 'Init Bwd Win Byts', 'Fwd Act Data Pkts',
         'Fwd Seg Size Min']
list5 = ['Dst Port', 'Protocol', 'TotLen Fwd Pkts', 'Fwd Pkt Len Max',
         'Fwd Pkt Len Min', 'Fwd Pkt Len Mean', 'Fwd Pkt Len Std',
         'Bwd Pkt Len Max', 'Bwd Pkt Len Min', 'Bwd Pkt Len Mean',
         'Bwd Pkt Len Std', 'Bwd IAT Tot', 'Fwd Header Len', 'Pkt Len Min',
         'Pkt Len Max', 'Pkt Len Mean', 'Pkt Len Std', 'Pkt Len Var',
         'RST Flag Cnt', 'PSH Flag Cnt', 'ACK Flag Cnt', 'ECE Flag Cnt',
         'Down/Up Ratio', 'Pkt Size Avg', 'Fwd Seg Size Avg', 'Bwd Seg Size Avg',
         'Subflow Fwd Byts', 'Init Bwd Win Byts', 'Fwd Act Data Pkts',
         'Fwd Seg Size Min']
list6 = ['Flow Duration', 'TotLen Fwd Pkts', 'Fwd Pkt Len Max',
         'Fwd Pkt Len Min', 'Fwd Pkt Len Mean', 'Fwd Pkt Len Std',
         'Bwd Pkt Len Max', 'Bwd Pkt Len Min', 'Bwd Pkt Len Mean',
         'Bwd Pkt Len Std', 'Flow IAT Std', 'Fwd IAT Tot', 'Fwd IAT Std',
         'Pkt Len Min', 'Pkt Len Max', 'Pkt Len Mean', 'Pkt Len Std',
         'Pkt Len Var', 'RST Flag Cnt', 'ECE Flag Cnt', 'Down/Up Ratio',
         'Pkt Size Avg', 'Fwd Seg Size Avg', 'Bwd Seg Size Avg',
         'Subflow Fwd Byts', 'Fwd Act Data Pkts', 'Active Mean', 'Active Max',
         'Active Min', 'Idle Std']
list7 = ['Dst Port', 'Protocol', 'Fwd Pkt Len Max', 'Fwd Pkt Len Min',
         'Fwd Pkt Len Mean', 'Fwd Pkt Len Std', 'Bwd Pkt Len Min',
         'Flow IAT Mean', 'Flow IAT Max', 'Flow IAT Min', 'Fwd IAT Mean',
         'Fwd IAT Max', 'Fwd IAT Min', 'Bwd IAT Std', 'Fwd PSH Flags',
         'Fwd URG Flags', 'Fwd Pkts/s', 'Bwd Pkts/s', 'Pkt Len Max',
         'Pkt Len Mean', 'FIN Flag Cnt', 'SYN Flag Cnt', 'PSH Flag Cnt',
         'ACK Flag Cnt', 'CWE Flag Count', 'Pkt Size Avg', 'Fwd Seg Size Avg',
         'Init Fwd Win Byts', 'Init Bwd Win Byts', 'Idle Max']
list8 = ['Protocol', 'Flow Duration', 'Fwd Pkt Len Max', 'Fwd Pkt Len Min',
         'Fwd Pkt Len Mean', 'Fwd Pkt Len Std', 'Bwd Pkt Len Max',
         'Bwd Pkt Len Min', 'Flow IAT Mean', 'Flow IAT Max', 'Flow IAT Min',
         'Fwd IAT Tot', 'Fwd IAT Mean', 'Fwd IAT Max', 'Fwd IAT Min',
         'Fwd URG Flags', 'Pkt Len Min', 'Pkt Len Max', 'Pkt Len Mean',
         'Pkt Len Std', 'PSH Flag Cnt', 'CWE Flag Count', 'Pkt Size Avg',
         'Fwd Seg Size Avg', 'Bwd Seg Size Avg', 'Init Bwd Win Byts',
         'Fwd Seg Size Min', 'Idle Mean', 'Idle Max', 'Idle Min']
list9 = ['Protocol', 'Flow Duration', 'Fwd Pkt Len Min', 'Bwd Pkt Len Max',
         'Bwd Pkt Len Min', 'Bwd Pkt Len Mean', 'Bwd Pkt Len Std',
         'Flow IAT Mean', 'Flow IAT Std', 'Flow IAT Max', 'Fwd IAT Tot',
         'Fwd IAT Mean', 'Fwd IAT Max', 'Fwd IAT Min', 'Bwd IAT Tot',
         'Pkt Len Min', 'Pkt Len Max', 'Pkt Len Mean', 'Pkt Len Std',
         'Pkt Len Var', 'RST Flag Cnt', 'ACK Flag Cnt', 'ECE Flag Cnt',
         'Pkt Size Avg', 'Bwd Seg Size Avg', 'Init Bwd Win Byts',
         'Fwd Seg Size Min', 'Idle Mean', 'Idle Max', 'Idle Min']
list10 = ['Src Port', 'Dst Port', 'Protocol', 'Fwd Pkt Len Max',
          'Fwd Pkt Len Min', 'Fwd Pkt Len Mean', 'Fwd Pkt Len Std',
          'Bwd Pkt Len Min', 'Bwd Pkt Len Std', 'Flow IAT Mean', 'Flow IAT Max',
          'Flow IAT Min', 'Fwd IAT Mean', 'Fwd IAT Std', 'Fwd IAT Max',
          'Fwd IAT Min', 'Bwd IAT Tot', 'Fwd PSH Flags', 'Pkt Len Min',
          'Pkt Len Std', 'SYN Flag Cnt', 'RST Flag Cnt', 'ACK Flag Cnt',
          'ECE Flag Cnt', 'Fwd Seg Size Avg', 'Init Bwd Win Byts',
          'Fwd Seg Size Min', 'Idle Mean', 'Idle Max', 'Idle Min']
# combine all lists into one
all_features = list1 + list2 + list3 + list4 + \
    list5 + list6 + list7 + list8 + list9 + list10

# count the occurrences of each feature
feature_counts = {}
for feature in all_features:
    if feature not in feature_counts:
        feature_counts[feature] = 0
    feature_counts[feature] += 1

# sort the features by their priority
sorted_features = sorted(feature_counts.items(),
                         key=lambda x: x[1], reverse=True)

# extract the top 15 features
common_features = [f[0] for f in sorted_features[:25]]
# print(common_features)
for i in common_features:
    print(i)
