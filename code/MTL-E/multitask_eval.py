from multitask_data_collator import DataLoaderWithTaskname
# import nlp
import numpy as np
import torch
import transformers
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay, precision_score, recall_score, f1_score, fbeta_score
import pandas as pd
import json



def multitask_eval_fn(multitask_model, model_name, features_dict, batch_size=8):
    preds_dict = {}
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    if torch.cuda.is_available():    
        device = torch.device("cuda")
#     for task_name in ["hyperbole", "metaphor","irony","sarcasm"]:
    for task_name in ["hyperbole", "metaphor"]:
        true_list=[]
        pred_list=[]
        val_len = len(features_dict[task_name]["validation"])
        acc = 0.0
        for index in range(0, val_len, batch_size):

            batch = features_dict[task_name]["validation"][
                index : min(index + batch_size, val_len)
            ]["doc"]
            labels = features_dict[task_name]["validation"][
                index : min(index + batch_size, val_len)
            ]["target"]
            inputs = tokenizer(batch, max_length=512, padding=True)
            inputs["input_ids"] = torch.LongTensor(inputs["input_ids"])
            inputs["attention_mask"] = torch.LongTensor(inputs["attention_mask"])
#             inputs["token_type_ids"] = torch.LongTensor(inputs["token_type_ids"])
#             print(type(inputs["input_ids"]))
#             print(type(inputs["attention_mask"]))
#             print(type(inputs["token_type_ids"]))
#             print(inputs["input_ids"])
#             print(inputs["attention_mask"])
            logits = multitask_model(task_name, **inputs.to(device))[0]

            predictions = torch.argmax(
                torch.FloatTensor(torch.softmax(logits, dim=1).detach().cpu().tolist()),
                dim=1,
            )
            true_list.extend(list(np.array(labels)))
            pred_list.extend(list(np.array(predictions)))
            acc += sum(np.array(predictions) == np.array(labels))
        acc = acc / val_len
        print(f"Task name: {task_name}")
        
        print("---------------------------------Confusion Matrix------------------------------------")
        final_create_confusion_matrix = confusion_matrix(true_list, pred_list)
        final_confusion_matrix_df = pd.DataFrame(final_create_confusion_matrix)
        print(final_confusion_matrix_df)
        # Precision, Recall and F1 score calculation
        final_eval_metrics = classification_report(true_list, pred_list, output_dict=True)
        print(final_eval_metrics)
        
        if task_name=="hyperbole":
            with open("../../results/hyperbole.json", "r") as file:
                data = json.load(file)
            data.append(final_eval_metrics)
            with open("../../results/hyperbole.json", "w") as file:
                json.dump(data, file, indent=4)
            
        elif task_name=="metaphor":
            with open("../../results/metaphor.json", "r") as file:
                data = json.load(file)
            data.append(final_eval_metrics)
            with open("../../results/metaphor.json", "w") as file:
                json.dump(data, file, indent=4)
        
        print("---------------------------------Evaluation Metrics------------------------------------")
        final_eval_metrics_df = pd.DataFrame(final_eval_metrics).transpose()
        final_eval_metrics_df = final_eval_metrics_df.iloc[: , :-1]
        print(final_eval_metrics_df)

    """
    print(eval_dataloader.data_loader.collate_fn)
    preds_dict[task_name] = trainer.prediction_loop(
        eval_dataloader,
        description=f"Validation: {task_name}",
    )
    for x in eval_dataloader:
        print(x)
    # Evalute task
    nlp.load_metric("glue", name="rte").compute(
        np.argmax(preds_dict[task_name].predictions),
        preds_dict[task_name].label_ids,
    )"""
