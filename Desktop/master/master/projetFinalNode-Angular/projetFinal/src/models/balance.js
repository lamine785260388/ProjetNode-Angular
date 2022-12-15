module.exports = (sequelize, DataTypes) => {
    return sequelize.define('BALANCE', {
      id: {
        type: DataTypes.INTEGER,
        primaryKey: true,
        autoIncrement: true
      },
      montant: {
        type: DataTypes.FLOAT,
        
      },
     
     
     
      

    })
  }