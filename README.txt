
How to get the showdown server working

pip install poke-env
Configuring a Pokémon Showdown Server
Though poke-env can interact with a public server, hosting a private server is advisable for training agents due to performance and rate limitations on the public server.

Install Node.js v10+.

Clone the Pokémon Showdown repository and set it up:

git clone https://github.com/smogon/pokemon-showdown.git
cd pokemon-showdown
npm install

Stop-Process -Name "node" -ErrorAction SilentlyContinue
npm install pg --save-dev

cp config/config-example.js config/config.js
node pokemon-showdown start --no-security