import pandas as pd
import pymysql as pml
from ml_utils import *
from get_data import *

dataframe = pd.DataFrame(columns=['anime_id', 'title', 'genre', 'num_episodes', 'picture', 'score'])


def connect_to_db(host, user, key, database, port_name):
    """Returns a connection to a database."""

    try:
        conn = pml.connect(host=host, user=user, password=key, port=port_name, database=database)
    except pml.OperationalError as e:
        raise e
    else:
        print('Connected!')

    return conn


def create_table(curr):
    """Creates a table in a database based on sql commands."""

    create_table_command = (""" CREATE TABLE IF NOT EXISTS animes_app (
                            anime_id INT(255) PRIMARY KEY,
                            title VARCHAR(120) NOT NULL,
                            genre VARCHAR(120),
                            num_episodes INT(6),
                            picture VARCHAR(200) NOT NULL,
                            score FLOAT(3,2)
                        )""")

    curr.execute(create_table_command)

    return None


def check_if_anime_exists(curr, anime_id):
    """Check if an anime already exist in the table."""

    query = """SELECT anime_id FROM animes_app WHERE anime_id = %s """
    curr.execute(query, (anime_id,))

    return curr.fetchone() is not None


def update_row(curr, title, genre, num_episodes, picture, score, anime_id):
    """Update a row in a table of the database."""

    query = (""" UPDATE animes_app
                SET title = %s,             
                genre = %s,
                num_episodes = %s,
                picture = %s,
                score = %s
                WHERE anime_id = %s;
                """)

    vars_to_update = (title, genre, num_episodes, picture, score, anime_id)
    curr.execute(query, vars_to_update)

    return None


def update_db(curr, df):
    """Update the table in database with new data."""

    temp_df = pd.DataFrame(columns=df.columns)

    for i, row in df.iterrows():
        if check_if_anime_exists(curr, row['anime_id']):  # if the video already exists, it will update
            update_row(curr, row['title'], row['genre'], row['num_episodes'], row['picture'], row['score'],
                       row['anime_id'])
        else:  # The video doesn't exist, so it will be added to a temp df using append_from_df_to_db
            temp_df = temp_df.append(row)

    return temp_df


def insert_into_table(curr, anime_id, title, genre, num_episodes, picture, score):
    """Inserts a dataframe row in a database table."""

    insert = (""" INSERT INTO animes_app(anime_id,
                title,             
                genre,
                num_episodes,
                picture,
                score)
                VALUES(%s, %s, %s, %s ,%s, %s)
                """)
    row_to_insert = (anime_id, title, genre, num_episodes, picture, score)
    curr.execute(insert, row_to_insert)

    return None


def append_from_df_to_db(curr, df):
    """Appends all dataframes rows into a database table."""

    for i, row in df.iterrows():
        insert_into_table(curr, row['anime_id'], row['title'], row['genre'], row['num_episodes'], row['picture'],
                          row['score'])

    return None


def extract_data(curr):
    """Extracts all rows of a table in a database with mediatype is equal to tv or ova"""

    curr.execute(""" SELECT * 
                    FROM animes_app 
                    WHERE media_type = 'tv' 
                    OR media_type = 'ova' 
                    """)
    df = pd.DataFrame(curr.fetchall())

    return df


def check_table_exists(curr, tablename):
    """Checks if a table already exist in database"""

    curr.execute(""" SELECT COUNT(*)
                    FROM information_schema.tables
                    WHERE table_name = '{0}'
                    """.format(tablename.replace('\'', '\'\'')))

    if curr.fetchone()[0] == 1:
        return True

    return False


def update_data_db(curr, df):
    response = get_anime(500)
    df = get_anime_details(response, df)

    if not check_table_exists(curr, 'animes_app'):
        create_table(curr)

    score = []
    for i, row in df.iterrows():
        score.append(compute_prediction(row))

    df['score'] = score

    new_anime_df = update_db(curr, df)
    append_from_df_to_db(curr, new_anime_df)
    conn.commit()

    return None
