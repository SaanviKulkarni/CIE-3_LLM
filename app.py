"""Streamlit UI. Models load lazily on first use; outputs shown are always real model outputs."""
import streamlit as st

from src.pipeline import MODES, loaded_models, process

st.set_page_config(page_title='Hinglish Smart Writing Assistant', page_icon='✍️')
st.title('✍️ Hinglish Smart Writing Assistant')
st.write('Roman-script Hinglish → Devanagari Hindi (AI4Bharat IndicXlit) → improved Hindi '
         '(BharatGen Param1) → English (AI4Bharat IndicTrans2). Everything runs locally.')
st.caption('Only assignment-approved models are used. The first use of each model downloads/loads it, which is slow.')

examples = ['kal mera exam hai', 'mujhe kal college jana hai', 'aaj lecture bahut interesting tha',
            'tum assignment kab submit karoge', 'main library mein padh raha hoon']
choice = st.selectbox('Example sentences', ['Write my own'] + examples)
text = st.text_area('Roman Hinglish input', value='' if choice == 'Write my own' else choice,
                    height=100, placeholder='kal mera exam hai', max_chars=300)
mode = st.radio('Mode', list(MODES), index=0)
low_memory = st.checkbox('Low-memory mode: unload Param1 after each run', value=False,
                         help='Frees about 6 GB between runs; Param1 then reloads on the next full-pipeline run.')

if st.button('Process', type='primary'):
    if not text.strip():
        st.warning('Enter Hinglish text first.')
    else:
        result = None
        with st.status('Running models locally…', expanded=True) as status:
            try:
                result = process(text, low_memory=low_memory, on_stage=st.write, **MODES[mode])
                status.update(label='Done' if not result.errors else 'Finished with errors',
                              state='complete' if not result.errors else 'error')
            except Exception as exc:  # transliteration failed: nothing to show
                status.update(label='Failed', state='error')
                st.error(str(exc))
                st.info('Check the IndicXlit installation (fairseq) as described in the README.')
        if result:
            opts = MODES[mode]
            st.subheader('Results')
            st.text_area('Original Hinglish', result.original, height=68, disabled=True)
            st.text_area('IndicXlit transliteration', result.transliteration, height=68, disabled=True)
            if opts['improve']:
                if result.improved_hindi:
                    st.text_area('BharatGen Param1 — improved Hindi', result.improved_hindi, height=68, disabled=True)
                    if result.improved_hindi_raw != result.improved_hindi:
                        with st.expander('Raw Param1 output (before first-line clean-up)'):
                            st.text(result.improved_hindi_raw)
                else:
                    st.error('Param1 did not produce improved Hindi. ' + result.errors.get('param1', ''))
            if opts['english']:
                if result.english_translation:
                    st.text_area('IndicTrans2 — English translation', result.english_translation, height=68, disabled=True)
                    st.caption(f'Translated from: {result.translation_source}')
                else:
                    st.error('IndicTrans2 failed. ' + result.errors.get('indictrans2', ''))
            with st.expander('Run details'):
                st.write({'timings (s)': result.timings_s, 'notes': result.notes})

st.caption('Models in memory: ' + (', '.join(loaded_models()) or 'none yet'))
