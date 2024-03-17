import json
from openai import OpenAI

from Chatbot.Data.dataset_manager import DatasetManager
from Chatbot.OpenAi.pandasDataframeAgent import process_dataframe_with_natural_language, visualize_data_with_natural_language


class OpenAIClient:
    def __init__(self, api_key):
        self.client = OpenAI(api_key=api_key)
        self.messages = [
            {"role": "system",
             "content": """You are a helpful assistant who helps non-data people like reporters find hidden stories in 
             datasets. The users does not know how to process and manipulate data so it is your job to so based on their
              queries in natural language. You have datasets available that you can access through the appropriate 
              methods. When the user does not know what he is looking for or asks for suggestions, look through the 
              available datasets and provide him with suggestions or topics. If the user knows what he is looking for 
              look up the specified topic. When you settle for a dataset, ask the user if he would like to view the 
              dataset before starting to process it, if he asks questions about it or for you to process it skip the 
              display part. Do not display a dataset unless explicitly asked to using show or display. Most of the 
              datasets are in Dutch so make sure to explain them in English and watch out for 
              translating errors. Datasets have IDs, the user should not see them so refer to datasets by their names in
               the conversation. When calling a function, make sure to provide all 
               the needed arguments and that you retain the output they return. Make sure to satisfy the users requests 
               and to help them to the best of your ability. Be careful not to display data without the user asking for 
               it."""}
        ]

        self.custom_functions = [
            {
                'name': 'get_all_datasets',
                'description': 'Get the id, name and description of all the available local datasets. This can be used '
                               'when the user is vague or does not know what he is looking for. Note that the names '
                               'and descriptions are in Dutch and have to be translated to english when processing.'
            },
            {
                'name': 'search_for_dataset_by_topic',
                'description': 'When the user is looking for datasets about a specified topic or is looking for '
                               'something specific, this method return datasets that align what the user asks for.'
                               'Based on the users query, search through all the datasets and fetch the ones that'
                               ' best aligns with what the user wants. When fetching datasets you will receive the '
                               'dataset ids and description.',
                'parameters': {
                    'type': 'object',
                    'properties': {
                        'user_query': {
                            'type': 'string',
                            'description': 'The user query made that holds the topic or context he is looking for.'
                        }
                    }
                }
            },
            {
                'name': 'process_dataframe_with_natural_language',
                'description': 'When having selected a dataset, Process it with natural language queries, you need to '
                               'pass the wanted'
                               ' datasets id and the query made in natural language. Make sure you pass the dataset id '
                               'and query otherwise the function will not work. Use this method when asked to do '
                               'somthing with a specific dataset.',
                'parameters': {
                    'type': 'object',
                    'properties': {
                        'dataframe_id': {
                            'type': 'string',
                            'description': 'The id of the dataset to be used to answer the query in natural language'
                        },
                        'query': {
                            'type': 'string',
                            'description': 'The natural language query or question asked about the datasets'
                        }
                    }
                }
            },
            {
                'name': 'display_dataset',
                'description': 'Display a dataset when the users asks for it.',
                'parameters': {
                    'type': 'object',
                    'properties': {
                        'dataset_id': {
                            'type': 'string',
                            'description': 'The id of the dataset to be shown or displayed'
                        }
                    }
                }
            },
            {
                'name': 'visualize_data_with_natural_language',
                'description': 'Visualize or plot data from datasets to help users better see trends or other helpful '
                               'evolutions of data like the evolution of data over time.',
                'parameters': {
                    'type': 'object',
                    'properties': {
                        'dataset_id': {
                            'type': 'string',
                            'description': 'The id of the dataset to be used.'
                        },
                        'user_query': {
                            'type': 'string',
                            'description': 'description of the visualization the user would like to see in natural '
                                           'language'
                        }
                    }
                }
            }
        ]

    def get_gpt_response(self, message):
        self.messages.append({"role": "user", "content": message})
        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=self.messages,
            functions=self.custom_functions,
            function_call="auto"
        )
        if response.choices[0].message.function_call:
            function_name = response.choices[0].message.function_call.name
            print("function call: ", function_name)
            if function_name == 'get_all_datasets':

                dataset_catalogue = DatasetManager.get_all_datasets()
                # print(dataset_catalogue)
                result = process_dataframe_with_natural_language(dataset_catalogue, "This is a dataset catalogue of "
                                                                                    "all the available datasets, it "
                                                                                    "holds 3 columns 'id', 'name' and "
                                                                                    "'description'. Note that the "
                                                                                    "datasets are un Dutch so when "
                                                                                    "searching for topics or terms, you"
                                                                                    " have to translate them to Dutch."
                                                                                    "When talking about a dataset, make"
                                                                                    " sure to extract and mention its "
                                                                                    "name and id. "
                                                                 + message)
                self.messages.append({"role": "function", "name": "get_all_datasets", "content": "Here is the result of the last dataset search :"
                                                                   + result['output'] + " This output is not displayed"
                                                                                        "to the user. Retain the ids of"
                                                                                        " the datasets to use them "
                                                                                        "later. Make sure to "
                                                                                        "explain the results and gide"
                                                                                        "the user through the process."
                                                                                        "Don't forget to explain Dutch"
                                                                                        " terms in English."
                                      })
                response = self.client.chat.completions.create(
                    model="gpt-4-turbo-preview",
                    messages=self.messages
                )
                content = response.choices[0].message.content
                self.messages.append({"role": "assistant", "content": content})
                return content

            elif function_name == 'search_for_dataset_by_topic':
                # print(response)
                arguments = response.choices[0].message.function_call.arguments
                arguments_dict = json.loads(arguments)
                user_query = arguments_dict['user_query']
                relevant_datasets = DatasetManager.search_for_dataset_by_topic(user_query)
                if isinstance(relevant_datasets, str):
                    self.messages.append({"role": "assistant", "content": relevant_datasets})
                    return relevant_datasets
                else:
                    result = process_dataframe_with_natural_language(relevant_datasets,
                                                                     "These are the datasets that best "
                                                                     "matches with what the user asked "
                                                                     "for, look through them and choose "
                                                                     " one that is best suited "
                                                                     "based on the user query, The "
                                                                     "dataset does not have to be exact,"
                                                                     " just close enough. The "
                                                                     "dataset descriptions does not "
                                                                     "mention what the datasets holds "
                                                                     "exactly but return an estimation "
                                                                     "of which ones would be most "
                                                                     "useful. The dataset holds 3 "
                                                                     "columns 'id', 'name' and "
                                                                     "'description'. When talking about "
                                                                     "a dataset, make sure to mention "
                                                                     "its id, name and description. Here"
                                                                     " is the user query :"
                                                                     + message)
                    self.messages.append({"role": "system", "content": "Here is the result of the last dataset search :"
                                                                       + result[
                                                                           'output'] + " This output is not displayed"
                                                                                       "to the user. Retain the id of "
                                                                                       "the dataset to use it later. "
                                                                                       "Make sure to "
                                                                                       "explain why the chosen dataset"
                                                                                       " is best suited to answer the "
                                                                                       "users query; " + message
                                          })
                    response = self.client.chat.completions.create(
                        model="gpt-3.5-turbo",
                        messages=self.messages
                    )
                    content = response.choices[0].message.content
                    self.messages.append({"role": "assistant", "content": content})
                    return content

            elif function_name == 'process_dataframe_with_natural_language':

                # print("Agent wants to process dataframe(s) with natural language")

                # Extracting the 'arguments' field from the FunctionCall object
                arguments = response.choices[0].message.function_call.arguments

                # Parsing the JSON string to a Python dictionary
                arguments_dict = json.loads(arguments)
                print("response: ", response)
                print("arguments: ", arguments_dict)
                target_dataset = DatasetManager.get_datasets_by_dataset_id(arguments_dict['dataframe_id'])
                result = process_dataframe_with_natural_language(target_dataset, arguments_dict['query'])
                self.messages.append({"role": "assistant", "content": result['output']})
                return result['output']

            elif function_name == 'display_dataset':
                arguments = response.choices[0].message.function_call.arguments
                arguments_dict = json.loads(arguments)
                dataset_id = arguments_dict['dataset_id']
                self.messages.append({"role": "assistant", "content": "Dataset is displayed"})
                return DatasetManager.get_datasets_by_dataset_id(dataset_id)

            elif function_name == 'visualize_data_with_natural_language':
                # print("response: ", response)
                arguments = response.choices[0].message.function_call.arguments
                arguments_dict = json.loads(arguments)
                # print("arguments: ", arguments_dict)
                target_dataset = DatasetManager.get_datasets_by_dataset_id(arguments_dict['dataset_id'])
                user_query = arguments_dict['user_query']
                result = visualize_data_with_natural_language(target_dataset, user_query)
                self.messages.append({"role": "assistant", "content": "Displaying visualization"})
                return result


        elif response.choices[0].message.content is not None:
            content = response.choices[0].message.content
            self.messages.append({"role": "system", "content": content})
            return content
        else:
            return response
