import subprocess
from transformers import GPT2Tokenizer

# ------------------- Configuration -------------------
weights_folder = "/extra/csph/gpt2/gpt2_medium_bin"
gpt2_executable = "./gpt2_infer"

# ------------------- Load tokenizer -------------------
tokenizer = GPT2Tokenizer.from_pretrained("gpt2-medium")

def run_gpt2_infer(input_text5, num_tokens_to_generate):
    # 1. Encode input text into token IDs
    input_ids = tokenizer.encode(input_text)
    print(input_ids)
    
    # 2. Prepare comma-separated string for C++ executable
    token_str = ",".join(str(t) for t in input_ids)

    result = subprocess.run([gpt2_executable, token_str, str(num_tokens_to_generate)],
                            capture_output=True, text=True)
    
    if result.returncode != 0:
        print("Error running GPT2 inference:")
        print(result.stderr)
        return None
    
    # 4. Parse generated token IDs from stdout
    # Assuming your C++ prints: Generated token IDs: 464,582,3111,...
    lines = result.stdout.splitlines()
    gen_line = [l for l in lines if "Generated token IDs:" in l]
    if not gen_line:
        print("No generated tokens found in output")
        return None

    token_id_str = gen_line[0].split(":")[1].strip()
    generated_ids = [int(x) for x in token_id_str.split(",")]
    
    # 5. Decode token IDs back to text
    generated_text = tokenizer.decode(generated_ids)
    
    return generated_ids, generated_text

# ------------------- Example usage -------------------
if __name__ == "__main__":
    input_text = "In a distant future, humanity has expanded far beyond the confines of Earth, establishing colonies across dozens of star systems. Each colony varies dramatically depending on the environmental conditions of its planet, from icy tundras to desert plains, from lush tropical ecosystems to barren volcanic landscapes. Advanced artificial intelligences oversee the management of each settlement, coordinating resource distribution, scientific research, and interstellar trade. Citizens interact daily with autonomous robots that assist in education, healthcare, transportation, and maintenance, while also collaborating with human experts on complex scientific and engineering problems. Space exploration continues at an unprecedented pace, with humans sending unmanned probes and manned missions to uncharted systems, discovering new celestial bodies, anomalies, and sometimes signs of alien life. Diplomatic relations extend beyond human civilizations, including interactions with alien species whose cultures and technologies challenge every preconceived notion humanity has held. Governance is a hybrid of AI councils and elected human representatives, designed to maintain fairness, enforce laws across light-years, and prevent resource conflicts between colonies. Education emphasizes adaptability and survival skills, preparing individuals to live on multiple planets with diverse gravity, atmosphere, and environmental hazards. Scientific breakthroughs occur daily in areas such as quantum physics, faster-than-light travel, energy production, and biotechnology. Artists use advanced holographic and virtual reality systems to create immersive experiences, while philosophers and ethicists debate the moral implications of AI oversight and human expansion. Trade networks span entire star systems, relying on quantum communication channels to overcome light-speed delays, while fleets of autonomous cargo ships transport goods and materials across vast distances. Emergencies such as meteor impacts, supernovae, or alien encounters are managed by coordinated human and AI teams, using predictive simulations and rapid-response technologies to mitigate damage. Citizens document their lives, discoveries, and interactions across planets, building an interstellar repository of knowledge, culture, and experience. Over time, these records form a complex web of interlinked civilizations, technologies, and philosophies, shaping the trajectory of humanity as it adapts, survives, and thrives across the stars. Each colony contributes uniquely to the ever-growing collective understanding of the universe, its dangers, and its wonders, ensuring that future generations inherit a legacy of exploration, innovation, and cooperation that transcends individual worlds and unites humanity on a galactic scale."
    n_iterations = 1
    num_tokens_to_generate = 20
    for i in range(n_iterations):
        generated_ids, generated_text = run_gpt2_infer(input_text, num_tokens_to_generate)
        
        print("Generated token IDs:", generated_ids)
        print("Generated text:", generated_text)
