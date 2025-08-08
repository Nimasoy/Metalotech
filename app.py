import predictor as p
import gradio as gr

demo = gr.Interface(
    fn=p.predictor,
    inputs=["text"],
    outputs=["text"],
)

if __name__ == "__main__":
    demo.launch(share=True)
