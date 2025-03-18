
import gradio as gr
import pickle 
import numpy as np
import pandas as pd


def predict(battery_power, blue, clock_speed, dual_sim, fc, four_g,
       int_memory, m_dep, mobile_wt, n_cores, pc, px_height,
       px_width, ram, sc_h, sc_w,talk_time, three_g,
       touch_screen, wifi):
    tst = pd.DataFrame([[battery_power, blue, clock_speed, dual_sim, fc, four_g,
           int_memory, m_dep, mobile_wt, n_cores, pc, px_height,
           px_width, ram, sc_h, sc_w,talk_time, three_g,
           touch_screen, wifi]],
          columns=['battery_power', 'blue', 'clock_speed', 'dual_sim', 'fc', 'four_g',
                 'int_memory', 'm_dep', 'mobile_wt', 'n_cores', 'pc', 'px_height',
                 'px_width', 'ram', 'sc_h', 'sc_w', 'talk_time', 'three_g',
                 'touch_screen', 'wifi'])    
    filehandler = open("mobile.pkl", "rb")
    bm_loaded = pickle.load(filehandler)
    filehandler.close()
    return bm_loaded.predict(tst)[0] 
      


with gr.Blocks() as demo:
    with gr.Row():
      battery_power = gr.Number(label='battery_power')
      blue = gr.Number(label='blue')
      clock_speed = gr.Number(label='clock_speed')
      dual_sim = gr.Number(label='dual_sim')
      fc = gr.Number(label='fc')
    with gr.Row():
      four_g = gr.Number(label='four_g')
      int_memory = gr.Number(label='int_memory')
      m_dep = gr.Number(label='m_dep')
      mobile_wt = gr.Number(label='mobile_wt')
      n_cores = gr.Number(label='n_cores')
    with gr.Row():
      pc = gr.Number(label='pc')
      px_height = gr.Number(label='px_height')
      px_width = gr.Number(label='px_width')
      ram = gr.Number(label='ram')
      sc_h = gr.Number(label='sc_h')
    with gr.Row():
      sc_w = gr.Number(label='sc_w')
      talk_time = gr.Number(label='three_g')
      three_g = gr.Number(label='three_g')
      touch_screen = gr.Number(label='touch_screen')
      wifi = gr.Number(label='wifi')
    with gr.Row(): 
      price_range = gr.Text(label='Price range:') 
    with gr.Row():  
      button = gr.Button(value="Which price range?")
      button.click(predict,
            inputs=[battery_power, blue, clock_speed, dual_sim, fc, four_g,
                   int_memory, m_dep, mobile_wt, n_cores, pc, px_height,
                   px_width, ram, sc_h, sc_w,talk_time, three_g,
                   touch_screen, wifi],
            outputs=[price_range])



demo.launch()