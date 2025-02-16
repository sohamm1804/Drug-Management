import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Load the Drug Discovery Dataset directly
@st.cache_data
def load_drug_discovery_data():
    # Replace with your actual dataset path
    return pd.read_csv("E:\dataset\dacha\drugs_datas.csv")  # Update with the correct path

# Function to load inventory and supply chain data
@st.cache_data
def load_data(file):
    return pd.read_csv(file)

# Function to save data
def save_data(df, path):
    df.to_csv(path, index=False)

# Streamlit sidebar for navigation
st.sidebar.title("Drug Management System")
st.sidebar.info("Manage drug discovery and supply chain data.")
app_mode = st.sidebar.selectbox("Choose the App Mode", ["Drug Discovery", "Inventory & Supply Chain Management"])

# ---- Drug Discovery Mode ----
if app_mode == "Drug Discovery":
    st.title("Drug Discovery App")

    # Load the drug discovery dataset
    df = load_drug_discovery_data()

    # Define features and targets
    X = df[['Drug Name', 'Chemical Composition', 'Category']]
    y = df[['Substitute Drug', 'Side Effects']]

    # Remove rows with missing target values and corresponding features
    df.dropna(subset=['Substitute Drug', 'Side Effects'], inplace=True)
    X = df[['Drug Name', 'Chemical Composition', 'Category']]
    y = df['Substitute Drug']  # Only 'Substitute Drug' is needed for model training

    # Ensure no missing values in input features
    X['Drug Name'].fillna('', inplace=True)
    X['Chemical Composition'].fillna('', inplace=True)
    X['Category'].fillna('Unknown', inplace=True)  # Replace NaN in 'Category' with 'Unknown'

    # Define feature transformers
    text_features = ['Drug Name', 'Chemical Composition']
    categorical_features = ['Category']

    # Transformer to flatten the list
    def flatten_text_data(X):
        return [' '.join(map(str, x)) for x in X]  # Convert each row to a single string

    text_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='')),
        ('flatten', FunctionTransformer(flatten_text_data, validate=False)),  # Flatten text data
        ('tfidf', TfidfVectorizer())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('text', text_transformer, text_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )

    # Create a pipeline with preprocessor and classifier for predicting drugs
    pipeline_drug = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(random_state=42))
    ])

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the model to predict drugs
    pipeline_drug.fit(X_train, y_train)

    # Create a mapping for side effects
    drug_side_effects = df[['Substitute Drug', 'Side Effects']].drop_duplicates().set_index('Substitute Drug')['Side Effects'].to_dict()

    # User input for prediction
    st.sidebar.header('Input Features')
    drug_name = st.sidebar.text_input('Drug Name', value='')
    chemical_composition = st.sidebar.text_input('Chemical Composition (optional)', value='')
    category = st.sidebar.text_input('Category (optional)', value='')

    # Combine user inputs into a DataFrame
    input_features = {
        'Drug Name': [drug_name.strip() or ''],
        'Chemical Composition': [chemical_composition.strip() or ''],
        'Category': [category.strip() or 'Unknown']  # Default to 'Unknown' if no input is provided
    }

    input_df = pd.DataFrame(input_features)

    # Make prediction if any input is provided
    if any([drug_name, chemical_composition, category]):
        try:
            # Predict the substitute drug
            predicted_drug = pipeline_drug.predict(input_df)[0]

            # Predict the side effects based on the predicted drug
            predicted_side_effects = drug_side_effects.get(predicted_drug, 'Unknown side effects')

            # Display predictions with only the answer in a box
            st.markdown("### Predicted Substitute Drug")
            st.markdown(
                f"""
                <div style='border: 2px solid #4CAF50; padding: 10px; border-radius: 10px; background-color: #e6ffe6;'>
                    <p style='font-size: 20px; color: #1b5e20;'><strong>{predicted_drug}</strong></p>
                </div>
                """, unsafe_allow_html=True
            )

            st.markdown("### Predicted Side Effects")
            st.markdown(
                f"""
                <div style='border: 2px solid #4CAF50; padding: 10px; border-radius: 10px; background-color: #e6ffe6;'>
                    <p style='font-size: 18px; color: #1b5e20;'>{predicted_side_effects}</p>
                </div>
                """, unsafe_allow_html=True
            )
        except Exception as e:
            st.error(f"An error occurred while making the prediction: {e}")
    else:
        st.write('Please enter at least one feature to get a prediction.')

# ---- Inventory & Supply Chain Management Mode ----
elif app_mode == "Inventory & Supply Chain Management":
    st.title("Drug Inventory & Supply Chain Management")

    # File upload for dynamic loading
    uploaded_inventory_file = st.sidebar.file_uploader("Upload Drug Inventory CSV", type=["csv"])
    uploaded_supply_chain_file = st.sidebar.file_uploader("Upload Supply Chain CSV", type=["csv"])

    if uploaded_inventory_file is not None and uploaded_supply_chain_file is not None:
        # Load datasets
        inventory_df = load_data(uploaded_inventory_file)
        supply_chain_df = load_data(uploaded_supply_chain_file)

        # Sidebar navigation
        options = st.sidebar.radio("Navigation", ["View Inventory", "Update Inventory", "View Supply Chain", "Update Supply Chain"])

        # View Inventory Section
        if options == "View Inventory":
            st.title("Drug Inventory")
            st.write("Below is the current drug inventory:")
            st.dataframe(inventory_df)

            # Filtering Options
            st.subheader("Filter Inventory")
            drug_name_filter = st.text_input("Filter by Drug Name:")
            if drug_name_filter:
                filtered_inventory = inventory_df[inventory_df["Drug Name"].str.contains(drug_name_filter, case=False, na=False)]
                st.write(f"Filtered Results for '{drug_name_filter}':")
                st.dataframe(filtered_inventory)

        # Update Inventory Section
        elif options == "Update Inventory":
            st.title("Update Inventory")
            st.write("Edit the inventory information directly in the table below.")

            # Display editable DataFrame
            st.dataframe(inventory_df, use_container_width=True)

            # Form to add new inventory data
            st.subheader("Add New Inventory Data")
            with st.form("add_inventory_form"):
                new_drug_name = st.text_input("Drug Name")
                new_quantity = st.number_input("Quantity", min_value=0)
                new_expiry_date = st.date_input("Expiry Date")
                new_supplier = st.text_input("Supplier")
                submit_inventory = st.form_submit_button("Add Inventory")

                if submit_inventory:
                    # Append new data to the DataFrame
                    new_row = pd.DataFrame({
                        "Drug Name": [new_drug_name],
                        "Quantity": [new_quantity],
                        "Expiry Date": [new_expiry_date],
                        "Supplier": [new_supplier]
                    })
                    inventory_df = pd.concat([inventory_df, new_row], ignore_index=True)
                    save_data(inventory_df, "path_to_save_updated_inventory.csv")  # Save to a file
                    st.success("New inventory data added successfully!")

                    # Display updated inventory
                    st.write("Updated Inventory:")
                    st.dataframe(inventory_df)

            # Form to update existing inventory data
            st.subheader("Update Existing Inventory Data")
            with st.form("update_inventory_form"):
                update_drug_name = st.selectbox("Select Drug to Update:", inventory_df["Drug Name"].unique())
                updated_quantity = st.number_input("Updated Quantity:", min_value=0)
                updated_expiry_date = st.date_input("Updated Expiry Date")
                updated_supplier = st.text_input("Updated Supplier")
                submit_update_inventory = st.form_submit_button("Update Inventory")

                if submit_update_inventory:
                    # Update existing data
                    inventory_df.loc[inventory_df["Drug Name"] == update_drug_name, "Quantity"]
                    inventory_df.loc[inventory_df["Drug Name"] == update_drug_name, "Quantity"] = updated_quantity
                    inventory_df.loc[inventory_df["Drug Name"] == update_drug_name, "Expiry Date"] = updated_expiry_date
                    inventory_df.loc[inventory_df["Drug Name"] == update_drug_name, "Supplier"] = updated_supplier
                    save_data(inventory_df, "path_to_save_updated_inventory.csv")  # Save to a file
                    st.success(f"Updated inventory data for '{update_drug_name}' successfully!")

                    # Display updated inventory
                    st.write("Updated Inventory:")
                    st.dataframe(inventory_df)

        # View Supply Chain Section
        elif options == "View Supply Chain":
            st.title("Drug Supply Chain")
            st.write("Below is the current supply chain information:")
            st.dataframe(supply_chain_df)

            # Filtering Options
            st.subheader("Filter Supply Chain")
            location_filter = st.text_input("Filter by Location:")
            if location_filter:
                filtered_chain = supply_chain_df[supply_chain_df["Location"].str.contains(location_filter, case=False, na=False)]
                st.write(f"Filtered Results for location '{location_filter}':")
                st.dataframe(filtered_chain)

        # Update Supply Chain Section
        elif options == "Update Supply Chain":
            st.title("Update Supply Chain")
            st.write("Edit the supply chain information directly in the table below.")

            # Display editable DataFrame
            st.dataframe(supply_chain_df, use_container_width=True)

            # Form to add new supply chain data
            st.subheader("Add New Supply Chain Data")
            with st.form("add_supply_chain_form"):
                new_drug_name = st.text_input("Drug Name")
                new_manufacturer = st.text_input("Manufacturer")
                new_distributor = st.text_input("Distributor")
                new_pharmacy = st.text_input("Pharmacy")
                new_location = st.text_input("Location")
                new_timestamp = st.date_input("Timestamp")
                submit_supply_chain = st.form_submit_button("Add Supply Chain")

                if submit_supply_chain:
                    # Append new data to the DataFrame
                    new_row = pd.DataFrame({
                        "Drug Name": [new_drug_name],
                        "Manufacturer": [new_manufacturer],
                        "Distributor": [new_distributor],
                        "Pharmacy": [new_pharmacy],
                        "Location": [new_location],
                        "Timestamp": [new_timestamp]
                    })
                    supply_chain_df = pd.concat([supply_chain_df, new_row], ignore_index=True)
                    save_data(supply_chain_df, "path_to_save_updated_supply_chain.csv")  # Save to a file
                    st.success("New supply chain data added successfully!")

                    # Display updated supply chain
                    st.write("Updated Supply Chain:")
                    st.dataframe(supply_chain_df)

            # Form to update existing supply chain data
            st.subheader("Update Existing Supply Chain Data")
            with st.form("update_supply_chain_form"):
                update_drug_name = st.selectbox("Select Drug to Update:", supply_chain_df["Drug Name"].unique())
                updated_location = st.text_input("Updated Location")
                submit_update_supply_chain = st.form_submit_button("Update Supply Chain")

                if submit_update_supply_chain:
                    # Update existing data
                    supply_chain_df.loc[supply_chain_df["Drug Name"] == update_drug_name, "Location"] = updated_location
                    save_data(supply_chain_df, "path_to_save_updated_supply_chain.csv")  # Save to a file
                    st.success(f"Updated supply chain location for '{update_drug_name}' successfully!")

                    # Display updated supply chain
                    st.write("Updated Supply Chain:")
                    st.dataframe(supply_chain_df)

    else:
        st.sidebar.error("Please upload both the Drug Inventory and Supply Chain CSV files.")
