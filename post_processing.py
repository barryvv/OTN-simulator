import pandas as pd
import os


def post_processing(training_folder, testing_folder):
    training_df_1_hop = pd.read_csv(training_folder + '/pairing1.csv')
    training_df_2_hop = pd.read_csv(training_folder + '/pairing2.csv')
    training_df_3_hop = pd.read_csv(training_folder + '/pairing3.csv')

    training_df_1_hop_gt = training_df_1_hop[training_df_1_hop["gt"] == 1]
    training_df_2_hop_gt = training_df_2_hop[training_df_2_hop["gt"] == 1]
    training_df_3_hop_gt = training_df_3_hop[training_df_3_hop["gt"] == 1]

    gt_feature_list = []
    for row in training_df_1_hop_gt.itertuples():
        gt_feature_list.append(row[4:])
    for row in training_df_2_hop_gt.itertuples():
        gt_feature_list.append(row[5:])
    for row in training_df_3_hop_gt.itertuples():
        gt_feature_list.append(row[6:])
    gt_feature_list = list(set(gt_feature_list))

    for root, dirs, files in os.walk(testing_folder):
        directory_list = list(dirs)
        for d in directory_list:
            result_df = pd.read_csv(os.path.join(root, d) + "/result.csv")
            output_df = pd.DataFrame([], columns=result_df.columns)
            for f in ["/pairing1.csv", "/pairing2.csv", "/pairing3.csv"]:
                testing_df = pd.read_csv(os.path.join(root, d) + f)
                if testing_df.shape[0] == 0:
                    continue

                num_hop = int(f[8])
                result_num_hop_df = result_df.copy()
                for i in range(0, num_hop + 1):
                    result_num_hop_df = result_num_hop_df[result_num_hop_df["Hop " + str(i)].notnull()]
                    result_num_hop_df = result_num_hop_df.astype({"Hop " + str(i): int})
                for i in range(num_hop + 1, 4):
                    result_num_hop_df = result_num_hop_df[result_num_hop_df["Hop " + str(i)].isnull()]
                result_num_hop_df = result_num_hop_df.reset_index(drop=True)

                for i in range(0, num_hop + 1):
                    if not result_num_hop_df["Hop " + str(i)].equals(testing_df["Hop " + str(i)]):
                        raise Exception("Hop " + str(i) + " is not the same.")
                testing_df["score"] = result_num_hop_df["score"]
                testing_df = testing_df[testing_df["score"] > 0.5]

                index_list = []
                for row in testing_df.itertuples():
                    feature = row[num_hop + 3: -1]
                    if feature in gt_feature_list:
                        index_list.append(row.Index)

                testing_df = testing_df[testing_df.index.isin(index_list)]
                df = pd.DataFrame([])
                for i in range(0, num_hop + 1):
                    df['Hop ' + str(i)] = testing_df['Hop ' + str(i)]
                df['gt'] = testing_df["gt"]
                df['score'] = testing_df["score"]
                output_df = pd.concat([output_df, df], ignore_index=True)

            output_df.to_csv(os.path.join(root, d) + "/result2.csv", index=False)


def main():
    post_processing("outputs/all boards once", "outputs/testing B")


main()
