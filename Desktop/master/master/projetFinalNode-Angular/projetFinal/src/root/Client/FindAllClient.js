
const { Client } = require('../../db/sequelize')

  
module.exports = (app) => {
  app.get('/api/findAllClient', (req, res) => {
  
  
      Client.findAll({order:['nom_client' ]})
      .then(agence => {
        const message = 'La liste des Clients a bien été récupérée.'
        res.json({ message, data: agence })
      })
     .catch(error=>{
      const message="la liste des Clients n'est pas disponibles.Résseyer dans quelques instant!"
      res.status(500).json({message,data:error})
     })
    
   
  })
}