list1 = ['Protocol', 'Fwd Pkt Len Max', 'Fwd Pkt Len Min', 'Fwd Pkt Len Mean',
         'Bwd Pkt Len Max', 'Bwd Pkt Len Min', 'Fwd Pkts/s', 'Bwd Pkts/s',
         'Pkt Len Min', 'Pkt Len Max', 'Pkt Len Mean', 'PSH Flag Cnt',
         'ACK Flag Cnt', 'URG Flag Cnt', 'Pkt Size Avg', 'Fwd Seg Size Avg',
         'Bwd Seg Size Avg', 'Init Fwd Win Byts', 'Fwd Act Data Pkts',
         'Fwd Seg Size Min']
list2 = ['Protocol', 'Flow Duration', 'Fwd Pkt Len Min', 'Bwd Pkt Len Max',
         'Bwd Pkt Len Min', 'Bwd Pkt Len Mean', 'Bwd Pkt Len Std', 'Fwd IAT Tot',
         'Pkt Len Min', 'Pkt Len Max', 'Pkt Len Mean', 'Pkt Len Std',
         'Pkt Len Var', 'RST Flag Cnt', 'ACK Flag Cnt', 'ECE Flag Cnt',
         'Pkt Size Avg', 'Bwd Seg Size Avg', 'Init Bwd Win Byts',
         'Fwd Seg Size Min']
list3 = ['Dst Port', 'TotLen Fwd Pkts', 'Fwd Pkt Len Max', 'Fwd Pkt Len Mean',
         'Fwd Pkt Len Std', 'Fwd Header Len', 'Pkt Len Max', 'Pkt Len Mean',
         'Pkt Len Std', 'Pkt Len Var', 'RST Flag Cnt', 'PSH Flag Cnt',
         'ACK Flag Cnt', 'ECE Flag Cnt', 'Pkt Size Avg', 'Fwd Seg Size Avg',
         'Subflow Fwd Byts', 'Init Fwd Win Byts', 'Init Bwd Win Byts',
         'Fwd Act Data Pkts']
list4 = ['Protocol', 'TotLen Fwd Pkts', 'Fwd Pkt Len Max', 'Fwd Pkt Len Mean',
         'Fwd Pkt Len Std', 'Bwd Pkt Len Max', 'Bwd Pkt Len Mean',
         'Fwd Header Len', 'Pkt Len Max', 'Pkt Len Mean', 'Pkt Len Std',
         'Pkt Len Var', 'RST Flag Cnt', 'ECE Flag Cnt', 'Pkt Size Avg',
         'Fwd Seg Size Avg', 'Bwd Seg Size Avg', 'Subflow Fwd Byts',
         'Fwd Act Data Pkts', 'Fwd Seg Size Min']
list5 = ['Protocol', 'Fwd Pkt Len Max', 'Fwd Pkt Len Min', 'Fwd Pkt Len Mean',
         'Bwd Pkt Len Max', 'Bwd Pkt Len Min', 'Fwd Pkts/s', 'Bwd Pkts/s',
         'Pkt Len Min', 'Pkt Len Max', 'Pkt Len Mean', 'PSH Flag Cnt',
         'ACK Flag Cnt', 'URG Flag Cnt', 'Pkt Size Avg', 'Fwd Seg Size Avg',
         'Bwd Seg Size Avg', 'Init Fwd Win Byts', 'Fwd Act Data Pkts',
         'Fwd Seg Size Min']
list6 = ['TotLen Fwd Pkts', 'Fwd Pkt Len Min', 'Fwd Pkt Len Mean',
         'Bwd Pkt Len Mean', 'Flow IAT Std', 'Fwd IAT Std', 'Pkt Len Min',
         'Pkt Len Mean', 'Pkt Len Std', 'Pkt Len Var', 'RST Flag Cnt',
         'ECE Flag Cnt', 'Pkt Size Avg', 'Fwd Seg Size Avg', 'Bwd Seg Size Avg',
         'Subflow Fwd Byts', 'Fwd Act Data Pkts', 'Active Mean', 'Active Min',
         'Idle Std']

# combine all lists into one
all_features = list1 + list2 + list3 + list4 + list5 + list6

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
common_features = [f[0] for f in sorted_features[:15]]
# print(common_features)
for i in common_features:
    print(i)
