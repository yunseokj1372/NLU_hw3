import sklearn.metrics
from transformers import RobertaModel

def compute_metrics(eval_pred):
    """Computes accuracy, f1, precision, and recall from a 
    transformers.trainer_utils.EvalPrediction object.
    """
    labels = eval_pred.label_ids
    preds = eval_pred.predictions.argmax(-1)

    ## TODO: Return a dictionary containing the accuracy, f1, precision, and recall scores.
    ## You may use sklearn's precision_recall_fscore_support and accuracy_score methods.
    dict = {}
    dict['accuracy_score'] = sklearn.metrics.accuracy_score(labels,preds)
    dict['f1_score'] = sklearn.metrics.f1_score(labels,preds)
    dict['precision_score'] = sklearn.metrics.precision_score(labels,preds)
    dict['recall_score'] = sklearn.metrics.recall_score(labels,preds)

    return dict
    

def model_init():
    """Returns an initialized model for use in a Hugging Face Trainer."""
    ## TODO: Return a pretrained RoBERTa model for sequence classification.
    ## See https://huggingface.co/transformers/model_doc/roberta.html#robertaforsequenceclassification.
    model = RobertaModel.from_pretrained("roberta-base")

    
