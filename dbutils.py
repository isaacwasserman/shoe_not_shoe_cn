import sqlite3
import json
import re


def add_dicts_to_db(db_name, table_name, dict_list, primary_key="id", update_duplicates=False):
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()

    # Get all columns and sanitize column names
    columns = set()
    for d in dict_list:
        columns.update(sanitize_column_name(col) for col in get_all_keys(d))

    columns = list(columns)

    # Ensure primary_key is in columns
    primary_key = sanitize_column_name(primary_key)
    if primary_key not in columns:
        columns.append(primary_key)

    # Create the table if it doesn't exist
    create_table_query = f"CREATE TABLE IF NOT EXISTS {table_name} (`{primary_key}` TEXT PRIMARY KEY)"
    cursor.execute(create_table_query)

    # Get existing columns
    cursor.execute(f"PRAGMA table_info({table_name})")
    existing_columns = set(row[1] for row in cursor.fetchall())

    # Add new columns if they don't exist
    for column in columns:
        if column not in existing_columns:
            alter_table_query = f"ALTER TABLE {table_name} ADD COLUMN `{column}` TEXT"
            cursor.execute(alter_table_query)

    # Prepare the insert or update query
    all_columns = list(existing_columns.union(columns))
    placeholders = ", ".join(["?" for _ in all_columns])
    column_names = ", ".join([f"`{col}`" for col in all_columns])

    if update_duplicates:
        update_clause = ", ".join([f"`{col}` = excluded.`{col}`" for col in all_columns if col != primary_key])
        query = f"""
        INSERT INTO {table_name} ({column_names})
        VALUES ({placeholders})
        ON CONFLICT(`{primary_key}`) DO UPDATE SET
        {update_clause}
        """
    else:
        query = f"""
        INSERT OR IGNORE INTO {table_name} ({column_names})
        VALUES ({placeholders})
        """

    # Insert or update the data
    for d in dict_list:
        flattened_dict = flatten_dict(d)
        sanitized_dict = {sanitize_column_name(k): v for k, v in flattened_dict.items()}
        values = [
            (
                json.dumps(sanitized_dict.get(col))
                if isinstance(sanitized_dict.get(col), (dict, list))
                else sanitized_dict.get(col, None)
            )
            for col in all_columns
        ]
        cursor.execute(query, values)

    conn.commit()
    conn.close()


def update_row_by_id(db_name, table_name, row_id, update_dict, primary_key="id"):
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()

    # Sanitize column names in the update dictionary
    sanitized_update = {sanitize_column_name(k): v for k, v in flatten_dict(update_dict).items()}

    # Get existing columns
    cursor.execute(f"PRAGMA table_info({table_name})")
    existing_columns = set(row[1] for row in cursor.fetchall())

    # Add new columns if they don't exist
    for column in sanitized_update.keys():
        if column not in existing_columns:
            alter_table_query = f"ALTER TABLE {table_name} ADD COLUMN `{column}` TEXT"
            cursor.execute(alter_table_query)

    # Prepare the update query
    update_clause = ", ".join([f"`{col}` = ?" for col in sanitized_update.keys()])
    query = f"UPDATE {table_name} SET {update_clause} WHERE `{primary_key}` = ?"

    # Prepare the values for the query
    values = list(sanitized_update.values()) + [row_id]

    # Execute the update query
    cursor.execute(query, values)

    # Commit the changes and close the connection
    conn.commit()
    conn.close()


# Example usage of update_row_by_id
# update_row_by_id("test.db", "test_table", "1", {"name": "New Name", "age": 30})


def count_rows_in_db(db_name, table_name):
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
    count = cursor.fetchone()[0]
    conn.close()
    return count


def get_all_columns(db_name, table_name):
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    cursor.execute(f"PRAGMA table_info({table_name})")
    columns = [row[1] for row in cursor.fetchall()]
    conn.close()
    return columns


def get_all_rows(db_name, table_name, columns=None):
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    if columns:
        columns = ", ".join(columns)
        cursor.execute(f"SELECT {columns} FROM {table_name}")
    else:
        cursor.execute(f"SELECT * FROM {table_name}")
    rows = cursor.fetchall()
    conn.close()
    return rows


def value_is_in_column(db_name, table_name, column_name, value):
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    cursor.execute(f"SELECT COUNT(*) FROM {table_name} WHERE {column_name} = ?", (value,))
    count = cursor.fetchone()[0]
    conn.close()
    return count > 0


def delete_by_id(db_name, table_name, id):
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    cursor.execute(f"DELETE FROM {table_name} WHERE id = ?", (id,))
    conn.commit()
    conn.close()


def get_all_keys(d, prefix=""):
    keys = []
    for k, v in d.items():
        new_key = f"{prefix}{k}" if prefix else k
        if isinstance(v, dict):
            keys.extend(get_all_keys(v, f"{new_key}_"))
        else:
            keys.append(new_key)
    return keys


def flatten_dict(d, prefix=""):
    flattened = {}
    for k, v in d.items():
        new_key = f"{prefix}{k}" if prefix else k
        if isinstance(v, dict):
            flattened.update(flatten_dict(v, f"{new_key}_"))
        else:
            flattened[new_key] = v
    return flattened


def sanitize_column_name(name):
    # Replace hyphens with underscores and remove any other non-alphanumeric characters
    return re.sub(r"[^\w]", "_", name)
