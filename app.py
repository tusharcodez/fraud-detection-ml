{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "7276c4e1-fe3e-445d-a231-df7e4332dc23",
   "metadata": {},
   "outputs": [],
   "source": [
    "import streamlit as st\n",
    "import joblib\n",
    "import numpy as np\n",
    "import pandas as pd\n",
    "\n",
    "# Load the trained model pipeline\n",
    "model = joblib.load(\"fraud_detection.pkl\")\n",
    "\n",
    "# App title\n",
    "st.title(\"💳 Fraud Detection System\")\n",
    "st.write(\"This app predicts whether a transaction is **fraudulent or genuine** based on input data.\")\n",
    "\n",
    "# Sidebar for user input\n",
    "st.sidebar.header(\"🔹 Enter Transaction Details\")\n",
    "\n",
    "# Collect user input\n",
    "oldbalanceOrg = st.sidebar.number_input(\"Old Balance (Source Account)\", min_value=0.0, step=100.0)\n",
    "newbalanceOrig = st.sidebar.number_input(\"New Balance (Source Account)\", min_value=0.0, step=100.0)\n",
    "oldbalanceDest = st.sidebar.number_input(\"Old Balance (Destination Account)\", min_value=0.0, step=100.0)\n",
    "newbalanceDest = st.sidebar.number_input(\"New Balance (Destination Account)\", min_value=0.0, step=100.0)\n",
    "amount = st.sidebar.number_input(\"Transaction Amount\", min_value=0.0, step=100.0)\n",
    "\n",
    "# Transaction type (if your model uses it)\n",
    "type_option = st.sidebar.selectbox(\"Transaction Type\", ['PAYMENT', 'TRANSFER', 'CASH_OUT', 'DEBIT', 'CASH_IN'])\n",
    "\n",
    "# Prepare input data for prediction\n",
    "# ⚠️ Make sure columns and order match what you trained your model with\n",
    "input_data = pd.DataFrame({\n",
    "    'type': [type_option],\n",
    "    'amount': [amount],\n",
    "    'oldbalanceOrg': [oldbalanceOrg],\n",
    "    'newbalanceOrig': [newbalanceOrig],\n",
    "    'oldbalanceDest': [oldbalanceDest],\n",
    "    'newbalanceDest': [newbalanceDest]\n",
    "})\n",
    "\n",
    "# Predict button\n",
    "if st.button(\"🔍 Predict Fraud\"):\n",
    "    prediction = model.predict(input_data)[0]\n",
    "    \n",
    "    if prediction == 1:\n",
    "        st.error(\"🚨 This transaction is **FRAUDULENT**!\")\n",
    "    else:\n",
    "        st.success(\"✅ This transaction looks **SAFE**.\")\n",
    "\n",
    "# Optional: show raw input\n",
    "st.write(\"### Input Summary\")\n",
    "st.write(input_data)\n"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python [conda env:base] *",
   "language": "python",
   "name": "conda-base-py"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.13.5"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
