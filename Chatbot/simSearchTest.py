from Data.similarity_search import get_most_similar_datasets
from Data.dataset_manager import DatasetManager
from Messages.message_handler import MessageHandler

# Call get_most_similar_datasets in your main loop
if __name__ == "__main__":
    # dataset = DatasetManager.read_datasets('../datasets')
    dataset_descriptions = DatasetManager.get_dataset_descriptions()

    while True:
        user_query = input("Enter your question: ")
        if user_query.lower() == 'exit':
            break

        detected_query = MessageHandler.translate_message(user_query)
        # print(dataset_descriptions[:5])

        gpt_response = get_most_similar_datasets(detected_query, dataset_descriptions)
        print("GPT-3 Response:", gpt_response)
