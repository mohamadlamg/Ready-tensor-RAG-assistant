from pydantic import BaseModel
from typing import Annotated, Literal, List
from operator import add
from pyjokes import get_joke
from langgraph.graph.state import StateGraph, CompiledStateGraph
from langgraph.graph import END
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv('GROQ_API_KEY')



class Joke(BaseModel):
    text: str
    category: str

class JokeState(BaseModel):
    jokes: Annotated[List[Joke], add] = []
    critic: Annotated[List[int], add] = [] 
    jokes_choice: Literal["n", "c", "q"] = "n"
    category: str = "neutral"
    language: str = "en"
    quit: bool = False

def show_menu(state: JokeState) -> dict:
    # Afficher la dernière blague si elle existe
    if state.jokes:
        joke = state.jokes[-1]
        print(f"\n{'='*50}")
        print(f"🎭 BLAGUE ({joke.category.upper()}) 🎭")
        print(f"{'='*50}")
        print(f"{joke.text}")
        print(f"{'='*50}")
        # Pour déboguer
    
    print(f"\nCatégorie actuelle: {state.category}")
    user_input = input("[n] Next joke  [c] Change category  [q] Quit\n> ").strip().lower()
    
    # Validation de l'entrée
    if user_input not in ["n", "c", "q"]:
        print("Choix invalide, 'n' sélectionné par défaut.")
        user_input = "n"
    
    return {"jokes_choice": user_input}

# def fetch_joke(state: JokeState) -> dict:
#     try:
#         print(f"🔄 Récupération d'une blague ({state.category})...")
        
#         # Déboguer: afficher la blague brute
#         joke_text = get_joke(language=state.language, category=state.category)
       
        
#         # Vérifier si la blague est vide ou tronquée
#         if not joke_text or len(joke_text) < 10:
           
#             joke_text = "Why do programmers prefer dark mode? Because light attracts bugs! 🐛"
        
#         new_joke = Joke(text=joke_text, category=state.category)
#         return {"jokes": [new_joke]}
        
#     except Exception as e:
#         print(f"❌ Erreur lors de la récupération de la blague: {e}")
#         # Retourner une blague par défaut
#         default_jokes = [
#             "Why do programmers prefer dark mode? Because light attracts bugs! 🐛",
#             "How many programmers does it take to change a light bulb? None, that's a hardware problem! 💡",
#             "Why do Java developers wear glasses? Because they don't C#! 👓",
#             "What's a programmer's favorite hangout place? The Foo Bar! 🍺",
#             "Why did the programmer quit his job? He didn't get arrays! 📊"
#         ]
#         import random
#         default_joke = Joke(text=random.choice(default_jokes), category=state.category)
#         return {"jokes": [default_joke]}

def writer(state: JokeState) -> dict:
   model = ChatGroq(model="llama-3.1-8b-instant",
               temperature=0.7,
               api_key=api_key)
   prompt = ChatPromptTemplate.from_template("""
    You have to write funny jokes (one by one) based on {category}.
    When user ask joke , write only one but don't show same things more than 2 times and be sure it is fun                                       
    You do not have to show the sources and the jokes must be no longer than 2 sentences.
    Just write jokes , you do not have to care about user thinking since you know the category write jokes                                       
    You have to write very funny jokes.                                                                                
    
    """)
   chain = prompt | model | StrOutputParser()

   response = response = chain.invoke({"category": state.category})
   new_joke = Joke(text=response, category=state.category)
   return{
       "jokes": [new_joke]
   }

def critic_func(state:JokeState):
    model = ChatGroq(model="llama-3.1-8b-instant",
               temperature=0.7,
               api_key=api_key)
    if not state.jokes:
        return {"critic": [5]}
    
    last_joke = state.jokes[-1].text
    
    # Prompt simple et direct
    prompt_text = f"""
    Rate this joke from 1 to 10 (where 10 is very funny and 1 is not funny at all).
    Only respond with a single number between 1 and 10.
    
    Joke: {last_joke}
    
    Your rating:"""
    
    try:
        response = model.invoke(prompt_text)
        
        # Extraire le score de la réponse
        response_text = response.content if hasattr(response, 'content') else str(response)
        score = int(''.join(filter(str.isdigit, response_text))[:2] or "5")
        score = max(1, min(10, score))
        
        print(f"📊 Score attribué: {score}/10")
        
    except Exception as e:
        print(f"⚠️ Erreur lors de l'évaluation: {e}")
        score = 5
    
    return {"critic": [score]}


def update_category(state: JokeState) -> dict:
    categories = ["neutral", "chuck", "all"]
    
    for i, cat in enumerate(categories):
        print(f"{i}: {cat}")
    
    try:
        selection = int(input("Sélectionnez une catégorie (0-2): ").strip())
        if 0 <= selection < len(categories):
            new_category = categories[selection]
            
            return {"category": new_category}
        else:
            print("Sélection invalide, catégorie inchangée.")
            return {}
    except ValueError:
        print("Entrée invalide, catégorie inchangée.")
        return {}

def exit_bot(state: JokeState) -> dict:
    print(f"\nMerci d'avoir utilisé le bot à blagues !")
    print(f"Vous avez lu {len(state.jokes)} blague(s).")
    return {"quit": True}

def route_choice(state: JokeState) -> str:
    if state.quit:
        return "exit_bot"
    elif state.jokes_choice == "n":
        return "writer"
    elif state.jokes_choice == "c":
        return "update_category"
    elif state.jokes_choice == "q":
        return "exit_bot"
    return "exit_bot"

def validation(state: JokeState) -> str:
    # ✅ Correction de la logique
    if state.critic and state.critic[-1] >= 7:
        return "continue"
    else:
        print(f"😕 Blague pas assez drôle (note: {state.critic[-1] if state.critic else 'N/A'}/10)")
        return "continue" 

def should_continue(state: JokeState) -> str:
    return "continue" if not state.quit else "end"

def build_joke_graph() -> CompiledStateGraph:
    workflow = StateGraph(JokeState)

    workflow.add_node("show_menu", show_menu)
    workflow.add_node("writer", writer)
    workflow.add_node("critic", critic_func)
    workflow.add_node("update_category", update_category)
    workflow.add_node("exit_bot", exit_bot)

    workflow.set_entry_point("writer")  # Commencer par récupérer une blague

    # Logique de routage améliorée
    workflow.add_conditional_edges(
        "show_menu",
        route_choice,
        {
            "writer": "writer",
            "critic": "critic",
            "update_category": "update_category", 
            "exit_bot": "exit_bot",
        }
    )

    # Ajouter une condition pour continuer ou terminer
    workflow.add_edge("writer", "critic")
    workflow.add_edge("critic", "display_and_menu")
    workflow.add_conditional_edges(
        "critic",
        validation,
        {
            "continue": "show_menu" , 
            "end": END 
        }
    )
    
    workflow.add_conditional_edges(
        "update_category", 
        should_continue,
        {
            "continue": "writer",  # Récupérer une blague après changement de catégorie
            "end": END
        }
    )
    
    workflow.add_edge("exit_bot", END)

    return workflow.compile()

def main():
    print("🎭 Bienvenue dans le Bot à Blagues ! 🎭")
    graph = build_joke_graph()
    
    try:
        final_state = graph.invoke(JokeState(), config={"recursion_limit": 100})
        print("Session terminée.")
    except KeyboardInterrupt:
        print("\n\nSession interrompue par l'utilisateur.")
    except Exception as e:
        print(f"Erreur: {e}")

if __name__ == "__main__":
    main()