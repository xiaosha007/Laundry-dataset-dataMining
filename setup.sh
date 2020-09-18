mkdir -p ~/.streamlit_dm/

echo "\
[server]\n\
headless = true\n\
port = $PORT\n\
enableCORS = false\n\
\n\
" > ~/.streamlit_dm/config.toml
