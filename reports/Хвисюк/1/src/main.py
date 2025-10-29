from step1 import clean_columns
from step2 import drop_missing_price
from step3 import plot_price_distribution
from step4 import show_avg_price_by_suburb
from step5 import add_property_age
from step6 import encode_type_column

def main():
    filepath = 'Melbourne_housing.csv'

    df = clean_columns(filepath)
    df = drop_missing_price(df)
    plot_price_distribution(df)
    show_avg_price_by_suburb(df)
    df = add_property_age(df)
    df = encode_type_column(df)

if __name__ == "__main__":
    main()
