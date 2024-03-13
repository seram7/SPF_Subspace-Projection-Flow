import torch
import pandas as pd
import streamlit as st
import os

st.set_page_config(layout="wide")

st.title('OOD Detection Performance: CIFAR100 as In-Distribution')


result_root_dir = '../results'
dataset = 'cifar100'

# dynamically load result directories
l_config_dir = [m for m in os.listdir(result_root_dir) if os.path.isdir(os.path.join(result_root_dir, m)) and m.startswith(dataset)]

l_models = []
for mm in l_config_dir:
    l_run = os.listdir(os.path.join(result_root_dir, mm))
    l_models += [mm + '/' + r for r in l_run]

# read auc and fpr95
l_auc_data = []
l_fpr_data = []
for m in l_models:
    result_dir = os.path.join(result_root_dir, m)
    l_auc_files = sorted([f for f in os.listdir(result_dir) if f.startswith('ood') and f.endswith('auc.txt')])
    l_fpr_files = sorted([f for f in os.listdir(result_dir) if f.startswith('ood') and f.endswith('fpr.txt')])
    l_auc = [pd.read_csv(os.path.join(result_dir, f), header=None).values[0][0] for f in l_auc_files]
    l_fpr = [pd.read_csv(os.path.join(result_dir, f), header=None).values[0][0] for f in l_fpr_files]
    l_auc_name = [f.split('_')[1] for f in l_auc_files]
    l_fpr_name = [f.split('_')[1] for f in l_fpr_files]
    d_auc_data = dict(zip(l_auc_name, l_auc))
    d_auc_data['name'] = m
    d_fpr_data = dict(zip(l_fpr_name, l_fpr))
    d_fpr_data['name'] = m

    # read author file
    author_file = os.path.join(result_dir, 'author.txt')
    if os.path.exists(author_file):
        with open(author_file, 'r') as f:
            author = f.read().strip()
    else:
        author = '-' 
    d_auc_data['author']= author
    d_fpr_data['author']= author

    # read memo file
    memo_file = os.path.join(result_dir, 'memo.txt')
    if os.path.exists(memo_file):
        with open(memo_file, 'r') as f:
            memo = f.read().strip()
    else:
        memo = '-'
    d_auc_data['memo']= memo
    d_fpr_data['memo']= memo

    l_auc_data.append(d_auc_data)
    l_fpr_data.append(d_fpr_data)

# display
df_auc = pd.DataFrame(l_auc_data)
df_fpr = pd.DataFrame(l_fpr_data)
st.subheader('AUC')
st.dataframe(df_auc, use_container_width=True)

st.subheader('FPR95')
st.dataframe(df_fpr, use_container_width=True)
    

    


