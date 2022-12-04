import zipfile
import streamlit as st
import pandas as pd
import numpy as np
from sdv.evaluation import evaluate
from table_evaluator import load_data, TableEvaluator
from sdv.tabular import CTGAN, TVAE, CopulaGAN, GaussianCopula
from st_aggrid import AgGrid, DataReturnMode, GridUpdateMode, GridOptionsBuilder, JsCode

st.set_page_config(page_title="Genera dati sintetici online", page_icon="📈", layout='wide', initial_sidebar_state='auto')

st.markdown("<center><h1> Genera Dati Sintetici gratis e online <small><br> Powered by INTELLIGENZAARTIFICIALEITALIA.NET </small></h1>", unsafe_allow_html=True)
st.write('<p style="text-align: center;font-size:15px;" > <bold> Genera dati sintetici su misura per il tuo dataset, hai a disposizione diversi modelli e opzioni <bold>  </bold><p>', unsafe_allow_html=True)


# upload csv file not accept multiple filesS
uploaded_file = st.file_uploader("Carica il tuo dataset", type=['csv'], accept_multiple_files=False)

if uploaded_file is not None:
    dataset = pd.read_csv(uploaded_file)
    colonne = list(dataset.columns)
    options = st.multiselect("Seleziona le colonne che vuoi usare..",colonne,colonne)
    dataset = dataset[options]
    gb = GridOptionsBuilder.from_dataframe(dataset)

    #customize gridOptions
    gb.configure_default_column(groupable=True, value=True, enableRowGroup=True, aggFunc='sum', editable=True)
    gb.configure_grid_options(domLayout='normal')
    gridOptions = gb.build()
    
    try:
        with st.expander("VISUALIZZA e MODIFICA il DATASET"):
            grid_response = AgGrid(
            dataset, 
            gridOptions=gridOptions,
            enable_enterprise_modules=True,
            update_mode="MODEL_CHANGED",
            )
    
        with st.expander("VISUALIZZA delle STATISICHE di BASE"):
            st.write(dataset.describe())
    except:
        print("")
        
    st.markdown("", unsafe_allow_html=True)
    
    generatoriSinte = ["CTGAN","TVAE","CopulaGAN","GaussianCopula"]
    modello = st.selectbox("Seleziona il generatore di dati sintetici",generatoriSinte,2)
    model = ""
    primaryKey = ""

    #checkbox per chiedere se c'è una primary key
    primary_key = st.checkbox("C'è una primary key?")
    if primary_key:
        primaryKey = st.text_input("Inserisci il campo primary key",value="")
     
    categorical_columns = None
    anonimize = st.checkbox("Ci sono dei campi da Anonimizzare?")
    if anonimize:
        #list all columns
        columns = list(dataset.columns)
        #selectbox for select categorical columns
        categorical_columns = st.multiselect("Seleziona le colonne da anonimizzare",columns) 
     
    @st.cache(allow_output_mutation=True)   
    def get_model(modello):
        with st.spinner("Caricamento modello in corso...🤖"):
            if modello == "CTGAN":
                if primaryKey != "" and categorical_columns == None:
                    model = CTGAN(primary_key=primaryKey)
                elif primaryKey != "" and categorical_columns != None:
                    model = CTGAN(primary_key=primaryKey, anonymize_fields=categorical_columns)
                else:
                    model = CTGAN()
                    
            elif modello == "TVAE":
                if primaryKey != "" and categorical_columns == None:
                    model = TVAE(primary_key=primaryKey)
                elif primaryKey != "" and categorical_columns != None:
                    model = TVAE(primary_key=primaryKey, anonymize_fields=categorical_columns)
                else:
                    model = TVAE()
            elif modello == "CopulaGAN":
                if primaryKey != "" and categorical_columns == None:
                    model = CopulaGAN(primary_key=primaryKey)
                elif primaryKey != "" and categorical_columns != None:
                    model = CopulaGAN(primary_key=primaryKey, anonymize_fields=categorical_columns)
                else:
                    model = CopulaGAN()
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
        with st.expander("🤗 Modello CTGAN - Dettagli"):
            st.write("Il modello CTGAN si basa sul sintetizzatore di dati Deep Learning basato su GAN che è stato presentato alla conferenza NeurIPS 2020 dal documento intitolato Modeling Tabular data using Conditional GAN. Grazie a questo modello è possibile generare dati sintetici per un dataset tabulare.")
    elif modello == "TVAE":
        with st.expander("🤗 Modello TVAE - Dettagli"):
            st.write("Il modello TVAE si basa sul sintetizzatore di dati Deep Learning basato su VAE che è stato presentato alla conferenza NeurIPS 2020 dal documento intitolato Modeling Tabular data using Variational Autoencoders. Grazie a questo modello è possibile generare dati sintetici per un dataset tabulare.")            
    elif modello == "CopulaGAN":
        with st.expander("🤗 Modello CopulaGAN - Dettagli"):
            st.write("Il modello CopulaGAN è una variazione del modello CTGAN che sfrutta la trasformazione basata su CDF applicata dalle GaussianCopulas per semplificare l'attività del modello CTGAN sottostante di apprendimento dei dati.")
    elif modello == "GaussianCopula":
        with st.expander("🤗 Modello GaussianCopula - Dettagli"):
            st.write("Il modello GaussianCopula è basato su funzioni copula . In termini matematici, una copula è una distribuzione sul cubo unitario[0,1]dche è costruito da una distribuzione normale multivariata su Rdutilizzando la trasformazione integrale di probabilità. Intuitivamente, una copula è una funzione matematica che ci permette di descrivere la distribuzione congiunta di più variabili casuali analizzando le dipendenze tra le loro distribuzioni marginali.")

    newdata = None
    
    with st.form(key='my_form'):
        col1, col2 = st.beta_columns(2)
        num_rows = col1.number_input("Numero di righe dagenerare",min_value=10, max_value=100000, value=1000, step=1)
        num_epochs = col2.number_input("Numero di epoch",min_value=10, max_value=500, value=300, step=1)
        submit_button = st.form_submit_button(label='Genera')
        
    if submit_button:
        with st.spinner("Generazione dati in corso...🤖"):
            model.fit(dataset, epochs=num_epochs)
            newdata = model.sample(num_rows)
            st.success("Dati generati!")
            # use aggrid to display the data
            grid_response = AgGrid( 
                newdata,
                gridOptions=gridOptions,
                enable_enterprise_modules=True,
                update_mode="MODEL_CHANGED"
            )
            
            #save model and data generated
            model.save("Modello_"+modello+ "_IntelligenzaArtificialeItalia.pkl")
            newdata.to_csv("DatiGenerati_"+modello+ "_IntelligenzaArtificialeItalia.csv", index=False)
            #merge data generated with original dataset
            merged_data = pd.concat([dataset, newdata], ignore_index=True)
            merged_data.to_csv("DatiSintetici+Originali_"+modello+ "_IntelligenzaArtificialeItalia.csv", index=False)
            
            #create zip file wit zipfile 
            zipObj = zipfile.ZipFile('DatiSintetici_IntelligenzaArtificialeItalia.zip', 'w')
            zipObj.write("Modello_Allenato_"+modello+ "_IntelligenzaArtificialeItalia.pkl")
            zipObj.write("DatiGenerati_"+modello+ "_IntelligenzaArtificialeItalia.csv")
            zipObj.write("DatiSintetici+Originali_"+modello+ "_IntelligenzaArtificialeItalia.csv")
            zipObj.close()
            
            #download zip file with st.download_button
            st.download_button(
                label="Scarica i dati sintetici",
                data=open("DatiSintetici_IntelligenzaArtificialeItalia.zip", 'rb').read(),
                file_name="DatiSintetici_IntelligenzaArtificialeItalia.zip",
                mime="application/zip",
            )
                
            
            #evalation section
            with st.expander("Valutazione dei dati sintetici generati"):
                valutazione = evaluate(dataset, newdata)
                st.write("La valutazione dei dati sintetici generati è:" + str(valutazione))
                table_evaluator = TableEvaluator(dataset, newdata)
                st.write(table_evaluator.visual_evaluation())
            
            

st.text("")
st.text("")
st.text("")
st.text("")
st.write("Proprietà intellettuale di [Intelligenza Artificiale Italia © ](https://intelligenzaartificialeitalia.net)")
st.write("Hai un idea e vuoi realizzare un Applicazione Web Intelligente? contatta il nostro [Team di sviluppatori © ](mailto:python.ai.solution@gmail.com)")