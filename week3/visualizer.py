import networkx as nx
import matplotlib.pyplot as plt
from typing import List, Dict
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


class TokenPredictor:
    """
    Runs a local Hugging Face model and records, for each generated token,
    the probability the model gave it and the top alternatives.
    """

    def __init__(self, model_name: str):
        self.model_name = model_name
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch.float16 if device == "cuda" else torch.float32,
        ).to(device)
        self.model.eval()

    def predict_tokens(self, prompt: str, max_tokens: int = 100) -> List[Dict]:
        """
        Generate text token by token (greedy, so temperature 0) and track prediction probabilities.
        Returns list of predictions with top token and alternatives.
        """
        if self.tokenizer.chat_template:
            text = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
            )
        else:
            text = prompt
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=False,  # greedy: always pick the most likely token
                output_scores=True,  # keep the scores of every step
                return_dict_in_generate=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        new_ids = out.sequences[0, inputs["input_ids"].shape[1]:]
        predictions = []
        for step, token_tensor in enumerate(new_ids):
            token_id = token_tensor.item()
            if token_id == self.tokenizer.eos_token_id:
                break

            # Turn the raw scores of this step into probabilities
            probs = torch.softmax(out.scores[step][0].float(), dim=-1)
            top_probs, top_ids = torch.topk(probs, k=3)

            alternatives = [
                (self.tokenizer.decode([i.item()]), p.item())
                for i, p in zip(top_ids, top_probs)
                if i.item() != token_id
            ][:2]  # Keep top 2 alternatives

            predictions.append(
                {
                    "token": self.tokenizer.decode([token_id]),
                    "probability": probs[token_id].item(),
                    "alternatives": alternatives,
                }
            )

        return predictions


def create_token_graph(model_name: str, predictions: List[Dict]) -> nx.DiGraph:
    """
    Create a directed graph showing token predictions and alternatives.
    """
    G = nx.DiGraph()

    G.add_node("START", token=model_name, prob="START", color="lightgreen", size=4000)

    # First, create all main token nodes in sequence
    for i, pred in enumerate(predictions):
        token_id = f"t{i}"
        G.add_node(
            token_id,
            token=pred["token"],
            prob=f"{pred['probability'] * 100:.1f}%",
            color="lightblue",
            size=6000,
        )

        if i == 0:
            G.add_edge("START", token_id)
        else:
            G.add_edge(f"t{i - 1}", token_id)

    # Then add alternative nodes with a different y-position
    last_id = None
    for i, pred in enumerate(predictions):
        parent_token = "START" if i == 0 else f"t{i - 1}"

        # Add alternative token nodes slightly below main sequence
        for j, (alt_token, alt_prob) in enumerate(pred["alternatives"]):
            alt_id = f"t{i}_alt{j}"
            G.add_node(
                alt_id, token=alt_token, prob=f"{alt_prob * 100:.1f}%", color="lightgray", size=6000
            )

            # Add edge from main token to its alternatives only
            G.add_edge(parent_token, alt_id)
            last_id = parent_token

    last_id = f"t{len(predictions) - 1}" if predictions else "START"
    G.add_node("END", token="END", prob="100%", color="red", size=6000)
    G.add_edge(last_id, "END")

    return G


def visualize_predictions(G: nx.DiGraph, figsize=(14, 80)):
    """
    Visualize the token prediction graph with vertical layout and alternating alternatives.
    """
    plt.figure(figsize=figsize)

    # Create custom positioning for nodes
    pos = {}
    spacing_y = 5  # Vertical spacing between main tokens
    spacing_x = 5  # Horizontal spacing for alternatives

    # Position main token nodes in a vertical line
    main_nodes = [n for n in G.nodes() if "_alt" not in n]
    for i, node in enumerate(main_nodes):
        pos[node] = (0, -i * spacing_y)  # Center main tokens vertically

    # Position alternative nodes to left and right of main tokens
    for node in G.nodes():
        if "_alt" in node:
            main_token = node.split("_")[0]
            alt_num = int(node.split("_alt")[1])
            if main_token in pos:
                # Place first alternative to left, second to right
                x_offset = -spacing_x if alt_num == 0 else spacing_x
                pos[node] = (x_offset, pos[main_token][1] + 0.05)

    # Draw nodes
    node_colors = [G.nodes[node]["color"] for node in G.nodes()]
    node_sizes = [G.nodes[node]["size"] for node in G.nodes()]
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes)

    # Draw all edges as straight lines
    nx.draw_networkx_edges(G, pos, edge_color="gray", arrows=True, arrowsize=20, alpha=0.7)

    # Add labels with token and probability
    labels = {node: f"{G.nodes[node]['token']}\n{G.nodes[node]['prob']}" for node in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels, font_size=14)

    plt.title("Token prediction.")
    plt.axis("off")

    # Adjust plot limits to ensure all nodes are visible
    margin = 8
    x_values = [x for x, y in pos.values()]
    y_values = [y for x, y in pos.values()]
    plt.xlim(min(x_values) - margin, max(x_values) + margin)
    plt.ylim(min(y_values) - margin, max(y_values) + margin)

    # plt.tight_layout()
    return plt
