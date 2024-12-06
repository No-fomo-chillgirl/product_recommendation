import streamlit as st
import pandas as pd
import pickle

# function cần thiết
def preprocess_text(text, stop_words):
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = re.sub(r'\d+', '', text)  # Remove digits
    text = text.lower().split()  # Lowercase and tokenize
    text = [word for word in text if word not in stop_words]  # Remove stopwords
    return text

# Function to get recommendations based on cosine similarity
def get_recommendations(df, ma_san_pham, cosine_sim, nums=5):
    # Ensure product IDs match the indices of cosine similarity matrix
    # Find the index of the selected product in df_products
    idx = df.index[df['ma_san_pham'] == ma_san_pham].tolist()
    
    if not idx:
        print(f"No product found with ID: {ma_san_pham}")
        return pd.DataFrame()  # Return an empty DataFrame
    
    idx = idx[0]  # Get the first index if there are multiple matches
    
    # Ensure that the index is within the bounds of the cosine similarity matrix
    if idx >= len(cosine_sim):
        print(f"Index {idx} is out of bounds for cosine similarity matrix.")
        return pd.DataFrame()  # Return an empty DataFrame if index is out of bounds
    
    # Get cosine similarity scores for the selected product
    sim_scores = list(enumerate(cosine_sim[idx]))
    
    # Sort products based on similarity scores in descending order
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    # Exclude the first product as it's the same one we selected
    sim_scores = sim_scores[1:nums+1]
    
    # Extract the indices of the most similar products
    product_indices = [i[0] for i in sim_scores]
    
    # Return the recommended products
    return df.iloc[product_indices]

def display_recommended_products(recommended_products, cols=5):
    for i in range(0, len(recommended_products), cols):
        columns = st.columns(cols)
        for j, col in enumerate(columns):
            if i + j < len(recommended_products):
                product = recommended_products.iloc[i + j]
                with col:
                    st.write(product['ten_san_pham'])
                    expander = st.expander("Mô tả")
                    truncated_description = ' '.join(product['mo_ta'].split()[:100]) + '...'
                    expander.write(truncated_description)

# Read data
df_products = pd.read_csv('Product.csv')
df_customer = pd.read_csv('Customer.csv')
df_output = pd.read_csv('output.csv')

# Load precomputed cosine similarity
with open('products_cosine_sim_.pkl', 'rb') as f:
    cosine_sim_new = pickle.load(f)

###### Streamlit Interface ######

# Display banner image
st.image('hasaki_banner_2.jpg', use_container_width=True)

# Step 1: Select a customer and suggest products based on their reviews
customer_options = [
    (row['ho_ten'], row['ma_khach_hang'])
    for _, row in df_customer[['ho_ten', 'ma_khach_hang']].drop_duplicates().iterrows()
]

selected_customer = st.selectbox( 
    "Chọn khách hàng",
    customer_options,
    format_func=lambda x: f"{x[0]} (Mã khách hàng: {x[1]})"
)

selected_customer_name, selected_customer_id = selected_customer
st.write(f"Khách hàng được chọn: **{selected_customer_name} (Mã: {selected_customer_id})**")

# Get the products this customer has interacted with from df_output
customer_products = df_output[df_output['ma_khach_hang'] == selected_customer_id]['ma_san_pham'].tolist()
if customer_products:
    st.write("### Sản phẩm đã mua:")
    purchased_products = df_products[df_products['ma_san_pham'].isin(customer_products)]
    st.write(purchased_products[['ma_san_pham', 'ten_san_pham']])

    st.write("### Gợi ý sản phẩm liên quan:")
    all_recommendations = pd.DataFrame()
    for product_id in customer_products:
        recommendations = get_recommendations(df_products, product_id, cosine_sim=cosine_sim_new, nums=3)
        recommendations = recommendations[recommendations['diem_trung_binh'] >= 4]
        all_recommendations = pd.concat([all_recommendations, recommendations])

    all_recommendations = all_recommendations.drop_duplicates(subset='ma_san_pham')
    all_recommendations = all_recommendations[~all_recommendations['ma_san_pham'].isin(customer_products)]
    display_recommended_products(all_recommendations, cols=3)
else:
    st.info(f"Khách hàng **{selected_customer_name}** chưa mua sản phẩm nào.")

# Step 2: Select a product and get similar product recommendations
st.write("---")
st.write("## Gợi ý sản phẩm tương tự")

if 'random_products' not in st.session_state:
    st.session_state.random_products = df_products.head(n=10)

product_options = [
    (row['ten_san_pham'], row['ma_san_pham'])
    for _, row in st.session_state.random_products.iterrows()
]

selected_product = st.selectbox(
    "Chọn sản phẩm",
    options=product_options,
    format_func=lambda x: x[0]
)

st.write("Bạn đã chọn:", selected_product)
st.session_state.selected_ma_san_pham = selected_product[1]

if st.session_state.selected_ma_san_pham:
    selected_product_df = df_products[df_products['ma_san_pham'] == st.session_state.selected_ma_san_pham]

    if not selected_product_df.empty:
        st.write('#### Bạn vừa chọn:')
        st.write('### ', selected_product_df['ten_san_pham'].values[0])

        product_description = selected_product_df['mo_ta'].values[0]
        truncated_description = ' '.join(product_description.split()[:100])
        st.write('##### Thông tin:')
        st.write(truncated_description, '...')

        st.write('##### Các sản phẩm liên quan:')
        recommendations = get_recommendations(
            df_products,
            st.session_state.selected_ma_san_pham,
            cosine_sim=cosine_sim_new,
            nums=3
        )
        display_recommended_products(recommendations, cols=3)
    else:
        st.write(f"Không tìm thấy sản phẩm với ID: {st.session_state.selected_ma_san_pham}")
