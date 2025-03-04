import argparse
import pickle 
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

############ Scikit-Learn Implementation ############
def sklearn_experiment(train_split, test_solar_ood_split, test_qwen_ood_split, test_llama_ood_split, ml_model, args):

    train_x = train_split['feature']
    solar_ood_text_x = test_solar_ood_split['feature']
    qwen_ood_test_x = test_qwen_ood_split['feature']
    llama_ood_test_x = test_llama_ood_split['feature']

    train_y = train_split['label']
    solar_ood_test_y = test_solar_ood_split['label']
    qwen_ood_test_y = test_qwen_ood_split['label']
    llama_ood_test_y = test_llama_ood_split['label']

    if ml_model == 'Logistic Regression':
        model = LogisticRegression(random_state=args.seed)

    scaler = StandardScaler()
    train_x = scaler.fit_transform(train_x)
    solar_ood_text_x = scaler.transform(solar_ood_text_x)
    qwen_ood_test_x = scaler.transform(qwen_ood_test_x)
    llama_ood_test_x = scaler.transform(llama_ood_test_x)

    # Training 
    model.fit(train_x, train_y)

    # Evaluation
    solar_ood_test_predictions = model.predict_proba(solar_ood_text_x)[:, 1]
    qwen_ood_test_predictions = model.predict_proba(qwen_ood_test_x)[:, 1]
    llama_ood_test_predictions = model.predict_proba(llama_ood_test_x)[:, 1]

    solar_ood_auroc = roc_auc_score(solar_ood_test_y, solar_ood_test_predictions)
    qwen_ood_auroc = roc_auc_score(qwen_ood_test_y, qwen_ood_test_predictions)
    llama_ood_auroc = roc_auc_score(llama_ood_test_y, llama_ood_test_predictions)
    
    # print results
    print("Solar OOD AUROC: ", solar_ood_auroc)
    print("Qwen OOD AUROC: ", qwen_ood_auroc)
    print("Llama OOD AUROC: ", llama_ood_auroc)

    total_results = {}
    solar_ood_results = {}
    solar_ood_results['label'] = solar_ood_test_y
    solar_ood_results['prediction'] = solar_ood_test_predictions
    qwen_ood_results = {}
    qwen_ood_results['label'] = qwen_ood_test_y
    qwen_ood_results['prediction'] = qwen_ood_test_predictions
    llama_ood_results = {}
    llama_ood_results['label'] = llama_ood_test_y
    llama_ood_results['prediction'] = llama_ood_test_predictions
    total_results['solar_ood'] = solar_ood_results
    total_results['qwen_ood'] = qwen_ood_results
    total_results['llama_ood'] = llama_ood_results

    return solar_ood_auroc, qwen_ood_auroc, llama_ood_auroc, total_results

#### Main Function ####
def main(args):

    print("Load " + args.text_type + "_" + str(args.seed) + " dataset...")
    # load dataset
    with open('preprocessed_data/' + args.text_type + '_' + str(args.seed) + '_ml_data.pkl', 'rb') as f:
        data = pickle.load(f)
    args.num_labels = 2
    train_dataset = data['train']
    test_solar_ood_dataset = data['human_solar_ood']
    test_qwen_ood_dataset = data['human_qwen_ood']
    test_llama_ood_dataset = data['human_llama_ood']

    # obtain text, feature, and label 
    train_text = []
    train_feature = []
    train_label = []
    test_solar_ood_text = []
    test_solar_ood_feature = []
    test_solar_ood_label = []
    test_qwen_ood_text = []
    test_qwen_ood_feature = []
    test_qwen_ood_label = []
    test_llama_ood_text = []
    test_llama_ood_feature = []
    test_llama_ood_label = []

    for inst in train_dataset:
        train_text.append(inst['text'])
        train_feature.append(inst['feature'])
        train_label.append(inst['label'])
    for inst in test_solar_ood_dataset:
        test_solar_ood_text.append(inst['text'])
        test_solar_ood_feature.append(inst['feature'])
        test_solar_ood_label.append(inst['label'])
    for inst in test_qwen_ood_dataset:
        test_qwen_ood_text.append(inst['text'])
        test_qwen_ood_feature.append(inst['feature'])
        test_qwen_ood_label.append(inst['label'])
    for inst in test_llama_ood_dataset:
        test_llama_ood_text.append(inst['text'])
        test_llama_ood_feature.append(inst['feature'])
        test_llama_ood_label.append(inst['label'])

    # Prepare Training and Test Data
    train_split = {}
    train_split['text'] = train_text
    train_split['feature'] = train_feature
    train_split['label'] = train_label

    test_solar_ood_split = {}
    test_solar_ood_split['text'] = test_solar_ood_text
    test_solar_ood_split['feature'] = test_solar_ood_feature
    test_solar_ood_split['label'] = test_solar_ood_label

    test_qwen_ood_split = {}
    test_qwen_ood_split['text'] = test_qwen_ood_text
    test_qwen_ood_split['feature'] = test_qwen_ood_feature
    test_qwen_ood_split['label'] = test_qwen_ood_label

    test_llama_ood_split = {}
    test_llama_ood_split['text'] = test_llama_ood_text
    test_llama_ood_split['feature'] = test_llama_ood_feature
    test_llama_ood_split['label'] = test_llama_ood_label

    ml_models = ['Logistic Regression']
    total_experiment_results = {}
    print("Conduct Experiments for ML Models with Comma Feature with Standard Scaling..")
    for ml_model in ml_models: 
        print("Conduct Experiment for " + ml_model + "..")
        solar_ood_auroc, qwen_ood_auroc, llama_ood_auroc, total_results = sklearn_experiment(train_split, test_solar_ood_split, test_qwen_ood_split, test_llama_ood_split, ml_model, args)
        result_item = {}
        result_item['solar_ood_auroc'] = solar_ood_auroc
        result_item['qwen_ood_auroc'] = qwen_ood_auroc
        result_item['llama_ood_auroc'] = llama_ood_auroc
        total_experiment_results[ml_model] = result_item
    with open(args.text_type + '_' + str(args.seed) + '_total_experiment_results.pkl', 'wb') as f:
        pickle.dump(total_experiment_results, f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--text_type', type=str, default="essay")
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()

    main(args)