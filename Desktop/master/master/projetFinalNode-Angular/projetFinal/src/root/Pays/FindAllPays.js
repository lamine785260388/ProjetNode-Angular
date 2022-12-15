
const { Pays, Devise } = require('../../db/sequelize')

  
module.exports = (app) => {
  app.get('/api/findAllPays', (req, res) => {
  
  
      Pays.findAll({order:['nom_pays']})
      .then(pays => {
        const message = 'La liste des Pays a bien été récupérée.'
        res.json({ message, data: pays })
      })
     .catch(error=>{
      const message="la liste des Pays n'est pas disponible.Résseyer dans quelques instant!"
      res.status(500).json({message,data:error})
     })
    
   
  })
}