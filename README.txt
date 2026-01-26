

How to get the showdown server working locally

pip install poke-env
Configuring a Pokémon Showdown Server
Though poke-env can interact with a public server, hosting a private server is advisable for training agents due to performance and rate limitations on the public server.

Install Node.js v10+.

Clone the Pokémon Showdown repository and set it up:

git clone https://github.com/smogon/pokemon-showdown.git
cd pokemon-showdown
npm install

Stop-Process -Name "node" -ErrorAction SilentlyContinue   Sometimes pg wont install so these 
npm install pg --save-dev                                 two lines can help fix that issue

cp config/config-example.js config/config.js
node pokemon-showdown start --no-security     //This runs Showdown on LocalHost



//to run in venv in root     .venv .\.venv\Scripts\Activate.ps1
//                           python -m pip install poke-env
//                           python -c "import poke_env; print(poke_env.__version__)"
// Run script example        python RandomPlayer.py     // Server has to be ran cd pokemon-showdown

