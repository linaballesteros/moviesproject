from django.shortcuts import render
import os
from dotenv import load_dotenv, find_dotenv
import json

import os
import openai

from openai.embeddings_utils import get_embedding, cosine_similarity
import numpy as np

from movie.models import Movie
# from movie import views



def recommend(request):

    env_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'openAI.env')
    load_dotenv(env_file_path)
    #  with open('movie_descriptions_embeddings.json', 'r') as archivo:
    openai.api_key = os.environ['openAI_api_key']

    json_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'movie_descriptions_embeddings.json')
    with open(json_file_path, 'r') as archivo:
        archivo_content = archivo.read()
        # archivo_content = file.write()
        movies = json.loads(archivo_content)
        
        # with open('movie_descriptions_embeddings.json', 'r') as archivo:

    req = request
    emb = get_embedding(req, engine='text-embedding-ada-002')

    sim = []
#
    for i in range(len(movies)):
        sim.append(cosine_similarity(emb,movies[i]['embedding']))
    sim = np.array(sim)
    idx = np.argmax(sim)

    
    pelicula = movies[idx]['title']
    return pelicula


def reco_home(request):

    req = request.POST.get('searchMovie')
    # req = request.POST.get('searchMovie')
    if req:
        title = recommend(req)
        movie = Movie.objects.filter(title__icontains = title) #     json_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'movie_descriptions_embeddings.json')

        return render(request, 'recommendations.html', {'searchMovie':req, 'movies': movie})

    else:
        """
        for i in range(len(movies)):
                sim.append(cosine_similarity(emb,movies[i]['embedding']))
            sim = np.array(sim)
            idx = np.argmax(sim)
    """
        return render(request, 'recommendations.html', {'movies': None})