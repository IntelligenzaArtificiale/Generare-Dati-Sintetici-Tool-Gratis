import os
import zipfile
import streamlit as st
from pandas_profiling import ProfileReport, compare
from streamlit_pandas_profiling import st_profile_report
import pandas as pd
import numpy as np
from sdv.evaluation import evaluate
from table_evaluator import load_data, TableEvaluator
from sdv.tabular import CTGAN, TVAE, CopulaGAN, GaussianCopula
from st_aggrid import AgGrid, DataReturnMode, GridUpdateMode, GridOptionsBuilder, JsCode

st.set_page_config(page_title="Genera dati sintetici online", page_icon="📈", layout='wide', initial_sidebar_state='auto')

st.markdown("<center><h1> Genera Dati Sintetici Gratis e Online <small><br> Powered by INTELLIGENZAARTIFICIALEITALIA.NET </small></h1>", unsafe_allow_html=True)
st.write('<p style="text-align: center;font-size:15px;" > <bold> Genera dati sintetici su misura per il tuo dataset, hai a disposizione diversi modelli e opzioni <bold>  </bold><p>', unsafe_allow_html=True)


# upload csv file not accept multiple filesS
uploaded_file = st.file_uploader("📤Carica i dati originali 📤", type=['csv'], accept_multiple_files=False)

if uploaded_file is not None:
    dataset = pd.read_csv(uploaded_file)
    #if dataset contains null or nan or empty or corrupted values drop them
    if dataset.isnull().values.any() or dataset.isna().values.any() or dataset.empty or dataset.isin(['']).values.any():
        dataset = dataset.dropna()
        dataset = dataset.replace('', np.nan)
        dataset = dataset.dropna()
        st.write("🚨 I dati originali contengono valori nulli, nan, vuoti o corrotti, sono stati eliminati 🚨")
    colonne = list(dataset.columns)
    options = st.multiselect("1️⃣ Seleziona le colonne che vuoi usare..",colonne,colonne)
    dataset = dataset[options]
    gb = GridOptionsBuilder.from_dataframe(dataset)

    #customize gridOptions
    gb.configure_default_column(groupable=True, value=True, enableRowGroup=True, aggFunc='sum', editable=True)
    gb.configure_grid_options(domLayout='normal')
    gridOptions = gb.build()
    
    try:
        with st.expander("📝 VISUALIZZA e MODIFICA il DATASET 📝"):
            grid_response = AgGrid(
            dataset, 
            gridOptions=gridOptions,
            enable_enterprise_modules=True,
            update_mode="MODEL_CHANGED",
            )
    
        with st.expander("📊 VISUALIZZA delle STATISICHE di BASE 📊"):
            st.write(dataset.describe())
                
    except:
        print("")
        
    st.markdown("", unsafe_allow_html=True)
    
    
    generatoriSinte = ["CTGAN","TVAE","CopulaGAN","GaussianCopula"]
    modello = st.selectbox("2️⃣ Seleziona il generatore di dati sintetici",generatoriSinte,2)
    col1, col2 = st.columns(2)
    num_row = col1.number_input("3️⃣ Numero di righe dagenerare",min_value=10, max_value=100000, value=200, step=1)
    num_epochs = col2.number_input("4️⃣ Numero di epoch",min_value=10, max_value=500, value=300, step=1)
    model = ""
    primaryKey = ""
    #list all columns
    columns = list(dataset.columns)
    
    #checkbox per chiedere se c'è una primary key
    primary_key = st.checkbox("🔑 C'è una primary key?")
    if primary_key:
        primaryKey = st.selectbox("🔐 Seleziona la primary key",columns)
    
    categorical_columns = None
   
    condizio = st.checkbox("🚫 Hai una condizione?")
    valore = "" 
    if condizio:
        cola, colb = st.columns(2)
        campoCondizione = cola.selectbox("🛂 Seleziona il campo condizione",columns)
        valore = colb.text_input("✅ Inserisci il valore",value="")
    
    @st.cache(allow_output_mutation=True)   
    def get_model(modello):
        with st.spinner("🤖Caricamento modello in corso...🤖"):
            if modello == "CTGAN":
                if primaryKey != "" and categorical_columns == None:
                    model = CTGAN(primary_key=primaryKey,epochs=num_epochs)
                elif primaryKey != "" and categorical_columns != None:
                    model = CTGAN(primary_key=primaryKey, anonymize_fields=categorical_columns,epochs=num_epochs)
                else:
                    model = CTGAN(epochs=num_epochs)
            elif modello == "TVAE":
                if primaryKey != "" and categorical_columns == None:
                    model = TVAE(primary_key=primaryKey,epochs=num_epochs)
                elif primaryKey != "" and categorical_columns != None:
                    model = TVAE(primary_key=primaryKey, anonymize_fields=categorical_columns,epochs=num_epochs)
                else:
                    model = TVAE(epochs=num_epochs)
            elif modello == "CopulaGAN":
                if primaryKey != "" and categorical_columns == None:
                    model = CopulaGAN(primary_key=primaryKey,epochs=num_epochs)
                elif primaryKey != "" and categorical_columns != None:
                    model = CopulaGAN(primary_key=primaryKey, anonymize_fields=categorical_columns,epochs=num_epochs)
                else:
                    model = CopulaGAN(epochs=num_epochs)
            elif modello == "GaussianCopula":
                if primaryKey != "" and categorical_columns == None:
                    model = GaussianCopula(primary_key=primaryKey)
                elif primaryKey != "" and categorical_columns != None:
                    model = GaussianCopula(primary_key=primaryKey, anonymize_fields=categorical_columns)
                else:
                    model = GaussianCopula()
                    
        return model
            
    model = get_model(modello)
    if modello == "CTGAN":
        with st.expander("🤗 Modello CTGAN - Dettagli🤖"):
            st.write("Il modello CTGAN si basa sul sintetizzatore di dati Deep Learning basato su GAN che è stato presentato alla conferenza NeurIPS 2020 dal documento intitolato Modeling Tabular data using Conditional GAN. Grazie a questo modello è possibile generare dati sintetici per un dataset tabulare.")
    elif modello == "TVAE":
        with st.expander("🤗 Modello TVAE - Dettagli🤖"):
            st.write("Il modello TVAE si basa sul sintetizzatore di dati Deep Learning basato su VAE che è stato presentato alla conferenza NeurIPS 2020 dal documento intitolato Modeling Tabular data using Variational Autoencoders. Grazie a questo modello è possibile generare dati sintetici per un dataset tabulare.")            
    elif modello == "CopulaGAN":
        with st.expander("🤗 Modello CopulaGAN - Dettagli🤖"):
            st.write("Il modello CopulaGAN è una variazione del modello CTGAN che sfrutta la trasformazione basata su CDF applicata dalle GaussianCopulas per semplificare l'attività del modello CTGAN sottostante di apprendimento dei dati.")
    elif modello == "GaussianCopula":
        with st.expander("🤗 Modello GaussianCopula - Dettagli🤖"):
            st.write("Il modello GaussianCopula è basato su funzioni copula . In termini matematici, una copula è una distribuzione sul cubo unitario[0,1]dche è costruito da una distribuzione normale multivariata su Rdutilizzando la trasformazione integrale di probabilità. Intuitivamente, una copula è una funzione matematica che ci permette di descrivere la distribuzione congiunta di più variabili casuali analizzando le dipendenze tra le loro distribuzioni marginali.")

    newdata = None


    
    if st.button("5️⃣ 👉🏻 🧬Genera Dati Sintetici Gratis🧬"):
        with st.spinner("🤖🧬Generazione Dataset Sintetico in corso...🤖🧬"):
            model.fit(dataset)
            newdata = None
            if condizio:
                from sdv.sampling import Condition
                condizione = Condition({
                    campoCondizione : valore
                }, num_rows=num_row )
                newdata = model.sample_conditions(conditions=[condizione])
            else:
                newdata = model.sample(num_row)
            st.success("🧬 Dati Sintetici generati! 🧬   - 🕗 Attendi un attimo mentre ti prepariamo i Report 🕗")
            # use aggrid to display the data
            with st.expander("📊🧬 Visualizza i dati generati"):
                grid_response = AgGrid( 
                    newdata,
                    gridOptions=gridOptions,
                    enable_enterprise_modules=True
                )
            
            #save model and data generated
            model.save("Modello_Allenato_"+modello+ "_IntelligenzaArtificialeItalia.pkl")
            newdata.to_csv("DatiGenerati_"+modello+ "_IntelligenzaArtificialeItalia.csv", index=False)
            #merge data generated with original dataset
            merged_data = pd.concat([dataset, newdata], ignore_index=True)
            merged_data.to_csv("DatiSintetici+Originali_"+modello+ "_IntelligenzaArtificialeItalia.csv", index=False)
            
            
            
            valutazione = evaluate(dataset, newdata)
            st.write("⚠ Il punteggio finale è un numero compreso tra 0 e 1, dove 0 indica la qualità più bassa e 1 quella più alta.")
            st.subheader("📊 La valutazione dei dati sintetici è : " + str(valutazione))

            try:
                with st.spinner("🤖🧬Generazione grafici differenze Dati Originali e Sintetici🤖🧬"):
                    with st.expander("📊 Grafici paragone dei dati sintetici con i dati originali"): 
                        #prendi il numero di righe casuali ugiuali al numero di righe del dataset sintetico
                        table_evaluator = TableEvaluator(dataset, newdata)
                        table_evaluator.visual_evaluation(save_dir="out")
                        #for all images in out folder st.image
                        for filename in os.listdir("out"):
                            st.image("out/"+filename)
                    
            except Exception as e:
                pass
                    
            with st.spinner("🔬⚗️ Genero il report dei Dati Originali... 🤖🧬"):
                with st.expander("🔍 SCARICA il report dei Dati Originali📕"):
                    pr = ProfileReport(dataset, title='Report Dati Originali', explorative=True)
                    pr.to_file("Report Dati Originali.html")
                    st.download_button("Scarica il report dei Dati Originali", open("Report Dati Originali.html", "rb").read(), mime="text/html")

            with st.spinner("🔬⚗️ Genero il report dei Dati Sintetici...🤖🧬"):
                with st.expander("🔍 SCARICA il report dei Dati Sintetici🧬"):
                    pr2 = ProfileReport(newdata, explorative=True, title="Report Dati Sintetici")
                    pr2.to_file("Report Dati Sintetici.html")
                    st.download_button("Scarica il report dei Dati Sintetici", open("Report Dati Sintetici.html", "rb").read(), mime="text/html")
            
            with st.spinner("🔬⚗️ Genero il report dei Dati Sintetici + Dati Originali...🧬🤖"):
                with st.expander("🔍 SCARICA il report dei Dati Sintetici🧬 + Dati Originali📕 "):
                    pr3 =  ProfileReport(merged_data, explorative=True, title="Report Dati Sintetici + Dati Originali")
                    pr3.to_file("Report Dati Sintetici + Dati Originali.html")
                    st.download_button("Scarica il report dei Dati Sintetici + Dati Originali", open("Report Dati Sintetici + Dati Originali.html", "rb").read(), mime="text/html")
                        
            with st.spinner("🔬⚗️Genero il report delle differze tra i dataset...🤖"):
                with st.expander("🔍 VISUALIZZA il report delle differze tra i dataset 📊📚"):
                    try:
                        comparison = compare([pr, pr2, pr3])
                        st_profile_report(comparison)
                        comparison.to_file("Report-Differenze.html")
                    except Exception as e:
                        print(e)
                        
            
            try:           
                #create zip file wit zipfile 
                zipObj = zipfile.ZipFile('DatiSintetici_IntelligenzaArtificialeItalia.zip', 'w')
                zipObj.write("Modello_Allenato_"+modello+ "_IntelligenzaArtificialeItalia.pkl")
                zipObj.write("DatiGenerati_"+modello+ "_IntelligenzaArtificialeItalia.csv")
                zipObj.write("DatiSintetici+Originali_"+modello+ "_IntelligenzaArtificialeItalia.csv")
                for filename in os.listdir("out"):
                    zipObj.write("out/"+filename)
                zipObj.write("Report Dati Originali.html")
                zipObj.write("Report Dati Sintetici.html")
                zipObj.write("Report Dati Sintetici + Dati Originali.html")
                zipObj.write("Report-Differenze.html")
                zipObj.close()
            except Exception as e:
                print(e)
                
            
            #download zip file with st.download_button
            st.download_button(
                label="📥📂 Scarica tutti i file 📑🎉",
                data=open("DatiSintetici_IntelligenzaArtificialeItalia.zip", 'rb').read(),
                file_name="DatiSintetici_IntelligenzaArtificialeItalia.zip",
                mime="application/zip",
            )
            st.success("📥🎉 Scarica tutti i file in un unico file zip! 📎📂")
            st.balloons()

st.text("")
st.text("")
st.text("")
st.text("")
st.write("Proprietà intellettuale di [Intelligenza Artificiale Italia © ](https://intelligenzaartificialeitalia.net)")
st.write("Hai un idea e vuoi realizzare un Applicazione Web Intelligente? contatta il nostro [Team di sviluppatori © ](mailto:python.ai.solution@gmail.com)")