
const { Agence } = require('../../db/sequelize')


  
module.exports = (app) => {
  app.get('/api/findAllAgence', (req, res) => {
  
  
      Agence.findAll({order:['nom_agence' ]})
      .then(agence => {
        const message = 'La liste des agences a bien été récupérée.'
        res.json({ message, data: agence })
      })
     .catch(error=>{
      const message="la liste des Agences n'est pas disponibles.Résseyer dans quelques instant!"
      res.status(500).json({message,data:error})
     })
    
   
  })
}