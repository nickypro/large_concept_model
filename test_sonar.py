# %%
from time import time as t
t0 = t()
import json
import torch
from sonar.inference_pipelines.text import TextToEmbeddingModelPipeline
from sonar.inference_pipelines.text import EmbeddingToTextModelPipeline

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE = torch.device(DEVICE)
torch.set_grad_enabled(False)
get_vram_used = lambda: torch.cuda.memory_allocated() / (1024 ** 2)  # Returns VRAM used in MB

time_taken = {}
time_taken["import modules"] = t() - t0

# %%
# Initialize the TextToEmbeddingModelPipeline
t0 = t()
v0 = get_vram_used()
t2vec_model    = TextToEmbeddingModelPipeline(encoder="text_sonar_basic_encoder", tokenizer="text_sonar_basic_encoder", device=DEVICE)
v1 = get_vram_used()
vec2text_model = EmbeddingToTextModelPipeline(decoder="text_sonar_basic_decoder", tokenizer="text_sonar_basic_encoder", device=DEVICE)
v2 = get_vram_used()

time_taken["load models"] = t() - t0

# %%
# TEST EXAMPLE
# 1. Get the SONAR embeddings
t0 = t()
sentences = ['My name is SONAR.', 'I can embed the sentences into vectorial space.']
embeddings = t2vec_model.predict(sentences, source_lang="eng_Latn")
print("Embeddings shape:", embeddings.shape)  # Should print something like: torch.Size([2, 1024])

# Step 2: Reconstruct text from SONAR embeddings
reconstructed = vec2text_model.predict(embeddings, target_lang="eng_Latn", max_seq_len=512)
print("Reconstructed text:", reconstructed)  # Should print the original sentences

time_taken["simple example"] = t() - t0

# %%
# REAL EXAMPLE
# Try with dataset examples
data = {"id": 11134, "prompt": "Write a news feed article titled \"G-Zero Experiment Spots Strange Quark Effects Not as Large as Hoped\", which covers the recent findings of the G-Zero experiment at Jefferson Lab, focusing on the implications of the experiment's results for our understanding of the strange quark and its influence on the proton, without delving into too much technical detail.", "completion": "**G-Zero Experiment Spots Strange Quark Effects Not as Large as Hoped**\n\nA team of physicists at Jefferson Lab has announced the results of the G-Zero experiment, a cutting-edge study aimed at understanding the behavior of strange quarks, a type of subatomic particle. While the experiment has made significant progress, the findings have also highlighted some intriguing, albeit modest, effects that challenge our current understanding of the strange quark's role in the proton.\n\nThe G-Zero experiment, which began in 2019, involved accelerating protons to nearly the speed of light and colliding them with an electromagnetic field. The goal was to study the strange quark's influence on the proton, a fundamental aspect of the Standard Model of particle physics. The researchers were hoping to observe a specific effect, known as the \"strange quark spin-flip,\" which would provide valuable insights into the strange quark's properties.\n\nAccording to the latest results, the experiment has indeed detected the spin-flip effect, but the magnitude of its influence on the proton's behavior is significantly smaller than initially anticipated. While this finding is a crucial step forward, it also raises questions about the role of the strange quark in the proton's dynamics.\n\n\"The strange quark's influence on the proton is still a topic of much debate among physicists,\" said Dr. Maria Rodriguez, lead researcher on the G-Zero experiment. \"Our results suggest that the strange quark's effect is more nuanced than we previously thought. It's as if the strange quark's influence is more subtle, but still present, in the proton's interactions.\"\n\nThe implications of this finding are far-reaching, as they challenge our current understanding of the strange quark's properties and its role in the proton. The strange quark is a mysterious particle, whose behavior is still not fully understood. This experiment's results highlight the need for further research into the strange quark's properties and its influence on the proton.\n\n\"While our results are not what we had hoped for, they are still a significant step forward in our understanding of the strange quark,\" said Dr. Rodriguez. \"This experiment will continue to run and gather data, and we are excited to see where this research will take us.\"\n\nThe G-Zero experiment is just one of several ongoing studies aimed at understanding the strange quark and its role in the proton. As the research continues, physicists hope to uncover more about the strange quark's properties and its influence on the fundamental forces of nature.\n\n**Related Stories:**\n\n* \"Jefferson Lab to Upgrade LANSCEP Facility for Next-Generation Particle Physics Research\"\n* \"Physicists Discover New Insights into Proton Structure with Latest Lattice QCD Simulations\"\n* \"G-Zero Experiment to Begin New Round of Data Collection Next Month\""}
#data = {"id": 12379, "prompt": "Write a Chapter titled \"The Game of Life: Exploring the Psychology of Competitive Engagement\", which delves into the intersection of personal identity, human values, and the drive to participate in competitive activities, including baseball.", "completion": "**Chapter 7: The Game of Life: Exploring the Psychology of Competitive Engagement**\n\nAs we walk through the gates of a baseball stadium, the crack of the bat, the smell of freshly cut grass, and the roar of the crowd immediately transport us to a world of intense competition. But what drives us to participate in such activities, and what do they reveal about our inner selves? The game of baseball, like many other competitive endeavors, is more than just a physical activity \u2013 it's a platform for exploring the intricacies of human psychology.\n\n**The Intersection of Personal Identity and Competitive Engagement**\n\nResearch in psychology has shown that our personal identity is intricately linked to our competitive engagement. For many individuals, participating in competitive activities is an essential aspect of their sense of self. When we compete, we are not just playing a game; we are testing our limits, demonstrating our skills, and asserting our identity as a player. This is particularly evident in the world of baseball, where players often have a strong sense of self and a deep connection to their team.\n\nTake, for example, the case of Derek Jeter, widely regarded as one of the greatest shortstops in baseball history. Jeter's on-field success was mirrored by his off-field persona \u2013 he was a natural leader, a clutch hitter, and a fierce competitor. His identity as a player was inextricably linked to his team, the New York Yankees, and his desire to win. For Jeter, competition was not just about winning games; it was about defining himself as a player and earning the respect of his peers.\n\n**Human Values and the Drive to Compete**\n\nCompetitive activities like baseball tap into fundamental human values, such as the desire for achievement, recognition, and belonging. For many individuals, participating in competitive activities provides a sense of purpose and direction, which can be particularly important during times of uncertainty or self-doubt.\n\nThe psychologist Mihaly Csikszentmihalyi, known as the \"Happiness Hypothesis,\" has extensively studied the concept of flow, a state of complete absorption and engagement in an activity. Flow experiences are characterized by heightened focus, concentration, and enjoyment, and are often associated with feelings of personal growth and accomplishment. In the context of baseball, flow experiences can be found in the heat of competition, as players become fully immersed in the game and lose track of time.\n\n**The Role of Emotions in Competitive Engagement**\n\nEmotions play a profound role in competitive engagement, influencing our motivations, behaviors, and overall experience. For many athletes, the thrill of competition is closely tied to the emotional highs and lows of winning and losing. The pressure to perform, the anxiety of failure, and the euphoria of success all contribute to a rich emotional landscape that is essential to the competitive experience.\n\nResearch has shown that emotions can also influence our self-perception and sense of identity. When we experience a strong emotion, such as pride or excitement, it can reinforce our personal identity and sense of self-worth. In the context of baseball, a player's emotional response to a game can be just as important as their physical performance. A player who is fully invested in the game, emotionally and mentally, is more likely to perform at their best.\n\n**The Psychology of Fan Engagement**\n\nBut the impact of competitive engagement extends far beyond the player's personal experience. The game of baseball has a profound influence on the fans, who are often deeply invested in the outcome of the game. Fans experience a range of emotions, from excitement and enthusiasm to anxiety and disappointment, which can be just as intense as those experienced by the players themselves.\n\nResearch has shown that fan engagement is closely tied to the sense of community and social identity that fans derive from attending games. Fans who feel a strong sense of belonging to a particular team or fan base are more likely to experience a sense of emotional connection to the game, which can lead to increased motivation and investment in the team's success.\n\n**Conclusion**\n\nThe game of baseball, like many other competitive endeavors, offers a unique window into the complexities of human psychology. Through our participation in competitive activities, we reveal aspects of our personal identity, including our values, motivations, and emotional responses. By exploring the psychology of competitive engagement, we can gain a deeper understanding of ourselves and our place in the world. Whether we are players, fans, or simply observers, the game of baseball continues to captivate and inspire us, offering a profound reflection of the human experience."}
original = [data["prompt"], *data["completion"].split("\n\n")]

# 1 - get embeds
t0 = t()
embeddings = t2vec_model.predict(original, source_lang="eng_Latn")
noise = torch.randn_like(embeddings)
print(embeddings.norm(dim=-1))
noise = 0.7 * embeddings.norm(dim=-1, keepdim=True) * noise / noise.norm(dim=-1, keepdim=True)
new_embeddings = embeddings + noise

# 2 - reconstruct outputs
reconstructed = vec2text_model.predict(new_embeddings, target_lang="eng_Latn", max_seq_len=512)
#reconstructed = vec2text_model.predict(embeddings, target_lang="eng_Latn", max_seq_len=512)
#print("Reconstructed text:", reconstructed)  # Should print the original
time_taken["dataset example"] = t() - t0

print(json.dumps([(original[i], reconstructed[i]) for i in range(len(reconstructed))] , indent=3))
print(embeddings.shape)
print(f"Importing modules took {time_taken['import modules']:.2f}s")
print(f"Models loaded in {time_taken['load models']:.2f}s, using {v2 - v0:.1f}MB VRAM ({v1-v0:.1f}MB enc + {v2-v1:.1f}MB dec)")
print(f"Simple example took {time_taken['simple example']:.2f}s")
print(f"Dataset example took {time_taken['dataset example']:.2f}s")
print(f"VRAM needed for context: {get_vram_used() - v2:.1f}MB")

# %%
get_pairwise_cossim = lambda emb : torch.nn.functional.cosine_similarity(emb.unsqueeze(1), emb.unsqueeze(0), dim=-1)
pairwise_cossim = get_pairwise_cossim(embeddings)
mean_cossim = pairwise_cossim[~torch.eye(pairwise_cossim.size(0), dtype=bool)].mean()
print("Mean cosine similarity:", mean_cossim.item())
print(json.dumps([str(i)+": "+x[:50] for (i, x) in enumerate(original)], indent=4))

print(torch.nn.functional.cosine_similarity(embeddings, new_embeddings))

# import seaborn as sns
# import matplotlib.pyplot as plt
# sns.heatmap(pairwise_cossim.cpu())
# plt.show()

# %%
text = """SONAR is a model from August 2023, trained as a semantic text auto-encoder, converting text into semantic embed vectors, which can later be then decoded back into text. Additionally, the model is trained such that the semantic embed vectors are to some degree "universal" for different languages, and one can embed in French and decode in English."""
# text = """I tried it, and SONAR seems to work surprisingly well. For example, the above paragraph and this paragraphs, if each are encoded into a two 1024 dimensional vectors (one for each paragraph), the model returns the following decoded outputs:"""
emb = t2vec_model.predict([text], source_lang="eng_Latn")
out = vec2text_model.predict(emb, target_lang="eng_Latn", max_seq_len=512)
print(text)
print(out)


# %%
from taker import Model
m = Model("meta-llama/Llama-3.2-3B-Instruct")

# %%

en = lambda: print("Is enabled:", m.hooks.collects["layer_0_pre_decoder"].enabled)
format_prompt = lambda _text : f"""<|start_header_id|>user<|end_header_id|>
{_text}<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
"""

en()
m.hooks.enable_collect_hooks("pre_decoder", layers=[0])
m.hooks.collects["layer_0_pre_decoder"].concat_mode = True
en()
m.generate(format_prompt("How many live in Japan?"), 50)
res = m.hooks["pre_decoder"]["collect"]
print(res[0].shape)

# %%
vec = res[0][0]
W_E = m.map["embed"].weight

token_ids = []
import torch
with torch.no_grad():
    for tok in vec:
        tok_id = ((tok.unsqueeze(0) - W_E)**2).sum(dim=-1).argmin(dim=-1)  # Shape: [26, 128256]
        token_ids.append(tok_id)
print(m.tokenizer.decode(token_ids))



# %%
