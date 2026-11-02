# SWYNEX AI Problem Design — Support Ticket Classifier

## Problem
Support teams receive tickets about billing, technical problems, orders/delivery, and account access. This prototype predicts a category to help route tickets to the appropriate team.

## Categories
- Payment and Billing
- Technical Issues
- Order and Delivery
- Account-Related Issues

## Dataset
`support_tickets.csv` contains 400 synthetic example messages (100 per category). It is intended for learning and prototyping—not production evaluation. Synthetic data can make results look better than performance on real customer messages.

## Approach
- TF-IDF word n-grams and character n-grams
- Logistic Regression classifier
- Stratified 75/25 train/test split
- Accuracy, macro F1, classification report, and confusion matrix
- Saves a model as `support_ticket_model.pkl`

## Setup
```bash
pip install -r requirements.txt
python train_classifier.py
```

Classify a message directly:
```bash
python train_classifier.py --message "I was charged twice for my subscription"
```

## Limitations
- Dataset is synthetic and template-generated.
- Test results are not a guarantee of real-world performance.
- Low-confidence predictions should be reviewed by a human.
- A production system needs real, permissioned, anonymized tickets; monitoring; and a human fallback.

## Evaluation
Report the metrics printed by the script. Do not claim a target score unless the measured test result actually reaches it.
