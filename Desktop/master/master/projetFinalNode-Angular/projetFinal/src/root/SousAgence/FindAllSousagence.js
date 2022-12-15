
const { SousAgence } = require('../../db/sequelize')

  
module.exports = (app) => {
  app.get('/api/findAllSousagence', (req, res) => {
  
  
      SousAgence.findAll({order:['nom_sous_agence' ]})
      .then(sousagence => {
        const message = 'La liste des sous agence a bien été récupérée.'
        res.json({ message, data: sousagence })
      })
     .catch(error=>{
      const message="la liste des sousagence n'est pas disponible.Résseyer dans quelques instant!"
      res.status(500).json({message,data:error})
     })
    
   
  })
}