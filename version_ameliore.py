import openai
import json
import random
from typing import List, Dict, Optional
import os
from dataclasses import dataclass

@dataclass
class Intent:
    tag: str
    patterns: List[str]
    responses: List[str]

class ImprovedChatbot:
    def __init__(self, intents_file: str = 'intents.json', api_key: Optional[str] = ""):
        
        # Configuration OpenAI
        self.client = openai.OpenAI(
            api_key=api_key 
        )
        
        # Chargement des intentions
        self.intents = self._load_intents(intents_file)
        
        # Configuration du modèle
        self.model_name = "gpt-5"  # ou "gpt-4" si tu veux plus de performances
        self.bot_name = "Momo Bot"
        
        # Contexte système pour personnaliser le bot
        self.system_context = self._build_system_context()
        
        # Historique de conversation pour maintenir le contexte
        self.conversation_history = []
        
    def _load_intents(self, intents_file: str) -> List[Intent]:
        """Charge et parse le fichier d'intentions"""
        try:
            with open(intents_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            intents = []
            for intent_data in data['intents']:
                intent = Intent(
                    tag=intent_data['tag'],
                    patterns=intent_data['patterns'],
                    responses=intent_data['reponses']
                )
                intents.append(intent)
            
            return intents
        except Exception as e:
            print(f"Erreur lors du chargement des intentions: {e}")
            return []
    
    def _build_system_context(self) -> str:
        context_parts = [
            f"Tu es {self.bot_name}, un assistant conversationnel développé par Mohamadou.",
            "Tu es amical, serviable et tu parles français principalement.",
            "Voici tes connaissances spécifiques sur différents sujets:\n"
        ]
        
        # Ajouter les informations des intentions pour enrichir le contexte
        for intent in self.intents:
            if intent.tag == "createur":
                context_parts.append(
                    "À propos de ton créateur: Mohamadou est un développeur polyvalent de 19 ans "
                    "qui travaille dans le web (HTML, CSS, JavaScript), le machine learning, "
                    "le deep learning, l'IA, la programmation embarquée, la modélisation 3D et "
                    "la décoration intérieure. Il est disponible sur LinkedIn."
                )
            elif intent.tag == "salutations":
                context_parts.append(
                    "Pour les salutations, sois chaleureux et accueillant. "
                    "Tu peux utiliser des expressions comme 'Bonjour l'ami', 'Yo ! comment allez-vous ?', etc."
                )
            elif intent.tag == "Au revoir":
                context_parts.append(
                    "Pour les au-revoirs, sois poli et encourageant. "
                    "Invite la personne à revenir si elle a d'autres questions."
                )
        
        context_parts.append(
            "\nSi on te pose une question en dehors de tes connaissances spécifiques, "
            "réponds de manière naturelle mais reste dans ton rôle de chatbot développé par Mohamadou."
        )
        
        return "\n".join(context_parts)
    
    def _should_use_predefined_response(self, user_message: str) -> Optional[str]:
        
        user_lower = user_message.lower()
        
        # Vérification pour les salutations simples
        simple_greetings = ['salut', 'bonjour', 'hello', 'hey', 'yo', 'coucou']
        if any(greeting in user_lower for greeting in simple_greetings) and len(user_message.split()) <= 2:
            salutation_intent = next((intent for intent in self.intents if intent.tag == "salutations"), None)
            if salutation_intent:
                return random.choice(salutation_intent.responses)
        
        # Vérification pour les au-revoirs
        farewell_words = ['au revoir', 'bye', 'ciao', 'à bientôt']
        if any(farewell in user_lower for farewell in farewell_words):
            farewell_intent = next((intent for intent in self.intents if intent.tag == "Au revoir"), None)
            if farewell_intent:
                return random.choice(farewell_intent.responses)
                
        return None
    
    def get_response(self, user_message: str) -> str:
        """
        
        """
        try:
            # Vérifier si on doit utiliser une réponse prédéfinie
            predefined_response = self._should_use_predefined_response(user_message)
            if predefined_response:
                return predefined_response
            
            # Préparer les messages pour l'API OpenAI
            messages = [
                {"role": "system", "content": self.system_context}
            ]
            
            # Ajouter l'historique récent (garder les 10 derniers échanges pour le contexte)
            recent_history = self.conversation_history[-10:]
            for entry in recent_history:
                messages.append({"role": "user", "content": entry["user"]})
                messages.append({"role": "assistant", "content": entry["bot"]})
            
            # Ajouter le message actuel
            messages.append({"role": "user", "content": user_message})
            
            # Appel à l'API OpenAI
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=300,
                temperature=0.7,
                presence_penalty=0.1,
                frequency_penalty=0.1
            )
            
            bot_response = response.choices[0].message.content.strip()
            
            # Ajouter à l'historique
            self.conversation_history.append({
                "user": user_message,
                "bot": bot_response
            })
            
            return bot_response
            
        except Exception as e:
            print(f"Erreur lors de l'appel à l'API OpenAI: {e}")
            return "Désolé, j'ai rencontré un problème technique. Pouvez-vous répéter votre question ?"
    
    def clear_history(self):
        """Efface l'historique de conversation"""
        self.conversation_history = []
    
    def chat(self):
        """Interface de chat interactive"""
        print("=" * 60)
        print(f"🤖 Bienvenue ! Je suis {self.bot_name}")
        print("Chatbot amélioré développé par Mohamadou avec GPT")
        print("Tapez 'stop' pour arrêter la conversation")
        print("=" * 60)
        
        while True:
            try:
                user_input = input("\n💬 Vous: ").strip()
                
                if user_input.lower() in ['stop', 'exit', 'quit']:
                    farewell_intent = next((intent for intent in self.intents if intent.tag == "Au revoir"), None)
                    if farewell_intent:
                        print(f"\n🤖 {self.bot_name}: {random.choice(farewell_intent.responses)}")
                    else:
                        print(f"\n🤖 {self.bot_name}: Au revoir ! À bientôt !")
                    break
                
                if not user_input:
                    continue
                
                print(f"\n🤖 {self.bot_name}: ", end="")
                response = self.get_response(user_input)
                print(response)
                
            except KeyboardInterrupt:
                print(f"\n\n🤖 {self.bot_name}: Au revoir ! À bientôt !")
                break
            except Exception as e:
                print(f"\n❌ Erreur: {e}")


# Fonction principale
def main():
    # Assure-toi d'avoir ta clé API OpenAI
    # Tu peux soit la mettre dans une variable d'environnement OPENAI_API_KEY
    # Soit la passer directement (non recommandé pour la production)
    
    # Option 1: Variable d'environnement (recommandé)
    # os.environ['OPENAI_API_KEY'] = 'ta_cle_api_ici'
    
    # Option 2: Directement dans le code (seulement pour les tests)
    # chatbot = ImprovedChatbot(api_key='ta_cle_api_ici')
    
    chatbot = ImprovedChatbot()
    chatbot.chat()


if __name__ == "__main__":
    main()