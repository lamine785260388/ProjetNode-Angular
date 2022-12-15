const express=require('express') 
const morgan=require('morgan')
const bodyParser=require('body-parser')
const port=3000
const app = express()
const  sequelize=require('./src/db/sequelize')
const auth = require('./src/auth/auth')
const cors=require('cors')

app
.use(bodyParser.json())
.use(morgan('dev'))
.use(cors())

sequelize.initDb()

require('./src/root/login')(app)
require('./src/root/createUser')(app)
require('./src/root/Client/createClient')(app)
require('./src/root/Devise/createDevise')(app)
require('./src/root/Pays/createPays')(app)
require('./src/root/Transaction/createTransaction')(app)
require('./src/root/Ville/createVille')(app)
require('./src/root/Agence/CreateAgence')(app)
require('./src/root/SousAgence/CreateSousAgence')(app)
require('./src/root/Paiement/CreatePaiement')(app)
require('./src/root/Balance/createBalance')(app)
require('./src/root/Agence/FindAllAgence')(app)
require('./src/root/Balance/FindAllBalance')(app)
require('./src/root/Client/FindAllClient')(app)
require('./src/root/Devise/FindAllDevise')(app)
require('./src/root/Paiement/FindAllPaiement')(app)
require('./src/root/Pays/FindAllPays')(app)
require('./src/root/SousAgence/FindAllSousagence')(app)
require('./src/root/Transaction/FindAllTransaction')(app)
require('./src/root/Ville/FindAllVille')(app)
require('./src/root/Agence/updateAgence')(app)
require('./src/root/Balance/updateBalance')(app)
require('./src/root/association/Pays_Devicefindone')(app)
//test middleware:



app.use(({res})=>{
     const message="impossible de trouver la ressource demandé! vous pouvez essayer une autre url"
    res.status(404).json({message})
    })
app.listen(port,()=>console.log('notre application node est démarré sur:http//localhost:'+port))