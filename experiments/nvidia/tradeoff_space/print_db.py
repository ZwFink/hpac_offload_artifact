import click
import sqlite3
import pandas as pd

@click.command()
@click.option('--input', help="Input database filename")
@click.option('--technique', help="Approx technique to print the output for. Must be one of: iact, taf, or perfo")
@click.option('--output', help="Output CSV Filename", default=None)
@click.option('--print_errors', help="Print the table of errors", default=False, is_flag=True)
def main(input, technique, output, print_errors):
    db = sqlite3.connect(input)
    cur = db.cursor()
    if print_errors:
        df = pd.read_sql_query("SELECT * FROM failures", db)
    else:
        df = pd.read_sql_query(f"SELECT * FROM {technique}", db)
    print(df)
    db.close()
    if output:
        df.to_csv(output, index=False)

if __name__ == '__main__':
    main()
