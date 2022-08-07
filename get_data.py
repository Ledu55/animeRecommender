# Imports

import time
import requests


def get_anime(number_of_anime):
    """Call myanimelist api and returns a list with anime attributes."""

    response = []
    for i in range(0, number_of_anime, 500):
        url = "https://api.myanimelist.net/v2/anime/ranking?fields=start_date, end_date, mean, genres, " \
              "media_type, status, num_episodes, rating&limit=500&offset={}".format(i)
        headers = {'X-MAL-CLIENT-ID': '3988e33ad8c7197c95ab720d9092f47c'}
        response.append(requests.get(url, headers=headers).json())
        time.sleep(1)

    return response


def get_anime_details(response, dataframe):
    """Returns a dataframe with cleaned data from response."""

    for items in response:
        for anime in items['data']:
            anime_id = anime['node']['id']
            title = anime['node']['title']
            if 'genres' in anime['node']:
                genre = anime['node']['genres'][0]['name']
            else:
                genre = ' '
            mean_score = anime['node']['mean']
            media_type = anime['node']['media_type']
            num_episodes = anime['node']['num_episodes']
            picture = anime['node']['main_picture']['medium']
            if 'rating' in anime['node']:
                rating = anime['node']['rating']
            else:
                rating = ' '
            status = anime['node']['status']
            if 'start_date' in anime['node']:
                start_date = anime['node']['start_date']
            else:
                start_date = ' '
            if 'end_date' in anime['node']:
                end_date = anime['node']['end_date']
            else:
                end_date = ' '

            # Save data in pandas dataframe
            dataframe = dataframe.append({'anime_id': anime_id,
                                          'title': title,
                                          'genre': genre,
                                          'mean_score': mean_score,
                                          'media_type': media_type,
                                          'num_episodes': num_episodes,
                                          'picture': picture,
                                          'rating': rating,
                                          'status': status,
                                          'start_date': start_date,
                                          'end_date': end_date}, ignore_index=True)
    return dataframe
