import sqlite3
import click

@click.command()
@click.option('--name', help="Name of the file to create")
def main(name):
    con = sqlite3.connect(name)
    cur = con.cursor()
    cur.execute('''CREATE TABLE perfo (exp_num integer, trial_num integer, items_per_thread integer, blocksize integer, num_blocks integer, num_threads integer, benchmark text, region text, error_metric text, perfo_type text, perfo_param int, runtime real, error real, start_time real, end_time real, hostname text)''')
    cur.execute('''CREATE TABLE iact (exp_num integer, trial_num integer, items_per_thread integer, blocksize integer, num_blocks integer, num_threads integer, benchmark text, region text, error_metric text, table_size integer, threshold real, tables_per_warp integer, replacement_policy text, hierarchy text, runtime real, error real, start_time real, end_time real, hostname text)''')
    cur.execute('''CREATE TABLE taf (exp_num integer, trial_num integer, items_per_thread integer, blocksize integer, num_blocks integer, num_threads integer, benchmark text, region text, error_metric text, threshold real, history_size integer, prediction_size integer, taf_width integer, hierarchy text, runtime real, error real, start_time real, end_time real, hostname text)''')
    cur.execute('''CREATE TABLE failures (exp_num integer, stderr text)''')
    cur.execute('''CREATE TABLE commands (exp_num integer, command text)''')
    con.commit()
    con.close()

if __name__ == '__main__':
    main()

