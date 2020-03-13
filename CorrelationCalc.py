# -*- coding: utf-8 -*-
"""
Created on Wed Mar 11 16:46:33 2020

@author: Matthew Rosencrans
"""
import numpy as np
import pandas as pd
testa10= pd.read_csv (r'C:\Users\Matthew Rosencrans\Desktop\CD10\test10.csv')
count10a0= pd.read_csv(r'C:\Users\Matthew Rosencrans\Desktop\CD10\model_count+0_cv0.csv')
count10a1= pd.read_csv(r'C:\Users\Matthew Rosencrans\Desktop\CD10\model_count+0_cv1.csv')
count10a2= pd.read_csv(r'C:\Users\Matthew Rosencrans\Desktop\CD10\model_count+0_cv0.csv')
count10a3= pd.read_csv(r'C:\Users\Matthew Rosencrans\Desktop\CD10\model_count+0_cv3.csv')
count10a4= pd.read_csv(r'C:\Users\Matthew Rosencrans\Desktop\CD10\model_count+0_cv4.csv')
count10a0.rename(columns={'pred':'binding_affinity'}, inplace=True)
count10a0.rename(columns={'':'AA_seq'}, inplace=True)