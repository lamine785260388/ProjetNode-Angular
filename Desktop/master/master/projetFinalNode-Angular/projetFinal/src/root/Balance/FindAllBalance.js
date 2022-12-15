
const { Balance } = require('../../db/sequelize')

  
module.exports = (app) => {
  app.get('/api/findAllBalance', (req, res) => {
  
  
      Balance.findAll()
      .then(agence => {
        const message = 'La liste des Balances a bien été récupérée.'
        
        res.json({ message, data: agence })
      })
     .catch(error=>{
      const message="la liste des Agences n'est pas disponibles.Résseyer dans quelques instant!"
      res.status(500).json({message,data:error})
     })
    
   
  })
}