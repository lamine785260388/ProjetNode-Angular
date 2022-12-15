
const { Devise } = require('../../db/sequelize')

  
module.exports = (app) => {
  app.get('/api/findAllDevise', (req, res) => {
  
  
      Devise.findAll({order:['nom_devise' ]})
      .then(agence => {
        const message = 'La liste des Devises a bien été récupérée.'
        res.json({ message, data: agence })
      })
     .catch(error=>{
      const message="la liste des Devises n'est pas disponibles.Résseyer dans quelques instant!"
      res.status(500).json({message,data:error})
     })
    
   
  })
}