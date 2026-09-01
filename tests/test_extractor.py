from src.extraction.extractor import extract_knowledge


# memory = "I built an LSTM using PyTorch for Tata Motors stock prediction."
#
# result = extract_knowledge(memory)
#
# print(result.model_dump_json(indent=2))

queries = [
    "I built a Tic Tac Toe neural network from scratch.",
    "I trained an LSTM model on Tata Motors stock data.",
    "I learned PyTorch for deep learning.",
    "I built a Flutter movie app using the TMDB API.",
    "I built a biometric attendance system using Arduino."
]

for q in queries:
    rslt = extract_knowledge(q)
    print(q)
    print(rslt.model_dump_json(indent=2))
    print('-----------------------------------------------------------------------------------------------')