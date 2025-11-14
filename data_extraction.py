import sys
import collections
import statistics
from collections import Counter

def read_data_with_filter(top_n=50):

    temp_data = collections.defaultdict(list)
    movie_counts = Counter()

    for line in sys.stdin:
        parts = line.strip().split()
        if len(parts) >= 3:
            user_id = int(parts[0])
            movie_id = int(parts[1])
            rating = float(parts[2])
            temp_data[user_id].append((movie_id, rating))
            movie_counts[movie_id] += 1

    top_movies = set(movie_id for movie_id, _ in movie_counts.most_common(top_n))

    filtered_r = collections.defaultdict(list)

    for user_id, ratings in temp_data.items():
        filtered_ratings = [(movie_id, rating) for movie_id, rating in ratings if movie_id in top_movies]
        if filtered_ratings:
            filtered_r[user_id] = filtered_ratings

    return filtered_r, top_movies, movie_counts


def main():
    # Read data and directly filter for top 50 movies
    filtered_r, top_movies, movie_counts = read_data_with_filter(top_n=100)
    user = {}
    try:
        with open("u.user", 'r') as file:
            for line in file:
                # Split by '|' character
                parts = line.strip().split('|')
                if len(parts) >= 4:
                    user_id = int(parts[0])
                    age = int(parts[1])
                    gender = parts[2]
                    occupation = parts[3]

                    user[user_id] = {
                        'age': age,
                        'gender': gender,
                        'occupation': occupation
                    }
    except FileNotFoundError:
        print(f"Error: File not found.")
    except Exception as e:
        print(f"Error reading user data: {e}")
    print(len(filtered_r.keys())) #number of items
    print(top_n) #number of platforms
    print(5) # number of groups
    print(2700) #utilityThreshold

    for i in filtered_r.keys():
        print(len(filtered_r[i]))
        for j,k in filtered_r[i]:
          print(j)
          print(k)

    for i in filtered_r.keys():
        print(user[i]["age"]//15)

    if not filtered_r:
        print("No data was read. Please provide valid input data.")
        return

if __name__ == "__main__":
    main()
    #Python3 data_extraction.py < u.data