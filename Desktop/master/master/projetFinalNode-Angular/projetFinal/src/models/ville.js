module.exports = (sequelize, DataTypes) => {
    return sequelize.define('VILLE', {
      id: {
        type: DataTypes.INTEGER,
        primaryKey: true,
        autoIncrement: true
      },
      code_iso2_ville:{
      type:DataTypes.STRING
      },
      nom_ville: {
        type: DataTypes.STRING,

      }
      
     
       

    })
  }